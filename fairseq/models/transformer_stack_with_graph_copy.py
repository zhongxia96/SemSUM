# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    MultiheadGraphAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from .transformer_with_copy import transformer_with_copyDecoder

@register_model('transformer_stack_with_graph_copy')
class GraphTransformerModel(FairseqModel):
    """
    stack the output of graph attn and self attn.
    head, tail representation. (two graph representation)
    ingoing, outgoing attention. (two graph attention)

    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (GraphTransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn', choices=['relu', 'gelu', 'gelu_fast'],
                            help='Which activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--graph-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

        parser.add_argument('--graph-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--graph-attention-dropout', type=float, metavar='D',
                            help='dropout probability for graph attention weights')

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict, edge_dict = task.source_dictionary, task.target_dictionary, task.edge_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            encoder_embed_edges = build_embedding(
                edge_dict, args.encoder_embed_dim, None
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
            encoder_embed_edges = build_embedding(
                edge_dict, args.encoder_embed_dim, None
            )

        encoder = GraphTransformerEncoder(args, src_dict, encoder_embed_tokens, encoder_embed_edges)
        decoder = transformer_with_copyDecoder(args, tgt_dict, decoder_embed_tokens)
        return GraphTransformerModel(encoder, decoder)

    def forward(self, src_tokens, src_lengths, enc_edge_ids, enc_edge_links1, enc_edge_links2, graph_mask, graph_mask_rev, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            enc_edge_ids (LongTensor): edges type in the source input of shape `(batch, edge_len)`
            enc_edge_links1 (LongTensor): edges head in the source input of shape `(batch, edge_len)`
            enc_edge_links2 (LongTensor): edges tail in the source input of shape `(batch, edge_len)`
            graph_mask && graph_mask_rev is the mask matrix with shape `(batch, src_len, edge_len)`
                e.g. we have tokens(query) 1,2,3,4 and edges(key) a,b,c, while adjacent link
                                                                                    is [(1,a,2), (2, b, 3), (2, c, 4) ].
                    the adjacent matrix as follows:
                                         a   b   c
                                    1   1    0   0
                                    2   1    1   1
                                    3   0    1   0
                                    4   0    0   1
                    for incoming relations, we have graph_mask as follows:
                                         a   b   c
                                    1   1    0   0
                                    2   0    1   1
                                    3   0    0   0
                                    4   0    0   0
                    for outgoing relations, we have graph_mask_rev as follows:
                                         a   b   c
                                    1   0    0   0
                                    2   1    0   0
                                    3   0    1   0
                                    4   0    0   1

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths, enc_edge_ids, enc_edge_links1, enc_edge_links2, graph_mask, graph_mask_rev)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class GraphTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, embed_edges):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.embed_dim = embed_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_edges = embed_edges
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.l1_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SELU(), nn.Dropout(self.dropout))
        self.l2_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SELU(), nn.Dropout(self.dropout))

        # self.e1_gate = nn.Linear(embed_dim*3, embed_dim)
        # self.e2_gate = nn.Linear(embed_dim * 3, embed_dim)

        self.e1_gate = nn.Sequential(nn.Linear(embed_dim*3, embed_dim), nn.SELU(), nn.Dropout(self.dropout))
        self.e2_gate = nn.Sequential(nn.Linear(embed_dim*3, embed_dim), nn.SELU(), nn.Dropout(self.dropout))

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        self.graph_layers = nn.ModuleList([])
        self.graph_layers.extend([
            GraphTransformerEncoderLayer(args)
            for i in range(args.graph_layers)
        ])

        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=False,
                           num_layers=1,
                           bidirectional=True)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
            self.graph_layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths, enc_edge_ids, enc_edge_links1, enc_edge_links2, graph_mask, graph_mask_rev):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        enc_edge_padding_mask = enc_edge_ids.eq(self.padding_idx)

        # embed edges
        enc_edge_emb = self.embed_scale * self.embed_edges(enc_edge_ids)
        enc_edge_emb = F.dropout(enc_edge_emb, p=self.dropout, training=self.training)

        # B x L x C -> L x B x C

        enc_edge_emb = enc_edge_emb.transpose(0, 1)

        # B x L -> L x B x C
        idx_pairs1 = enc_edge_links1.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        idx_pairs2 = enc_edge_links2.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        idx_pairs1 = idx_pairs1.transpose(0, 1)
        idx_pairs2 = idx_pairs2.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)



        # graph layers
        enc_states = []
        enc_states.append(x)

        for layer in self.graph_layers:

            x = layer(x, self.l1_gate, self.l2_gate, self.e1_gate, self.e2_gate, idx_pairs1, idx_pairs2, encoder_padding_mask, enc_edge_padding_mask, graph_mask, graph_mask_rev, enc_edge_emb)
            enc_states.append(x)

        enc_states = torch.stack(enc_states, dim=3)
        enc_states = torch.transpose(torch.transpose(enc_states, 2, 3), 0, 2)

        bsz = enc_states.size(1)
        seq_len = enc_states.size(2)

        enc_states = enc_states.reshape(len(self.graph_layers) + 1, bsz*seq_len, self.embed_dim)

        outputs, state = self.rnn(enc_states)

        x = state[0][::2] + state[1][::2]

        x = x.reshape(bsz, -1, self.embed_dim)
        x = torch.transpose(x, 0, 1)

        if self.normalize:
            x = self.graph_layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def reorder_encoder_input(self, encoder_input, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        # print('reorder')
        if encoder_input['src_tokens'] is not None:
            encoder_input['src_tokens'] = \
                encoder_input['src_tokens'].index_select(0, new_order)
        # print('reorder_finish')
        # if encoder_input['net_input'] is not None:
        #     encoder_input['net_input']['src_tokens'] = \
        #         encoder_input['net_input']['src_tokens'].index_select(0, new_order)
        return encoder_input

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, f"{name}.layers.{i}")

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class GraphTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        # self.self_attn = MultiheadAttention(
        #     self.embed_dim, args.encoder_attention_heads,
        #     dropout=args.attention_dropout,
        # )
        self.graph_attn = MultiheadGraphAttention(
            self.embed_dim, args.graph_attention_heads,
            dropout=args.graph_attention_dropout,
        )

        self.fusion_gate = nn.Sequential(nn.Linear(2 * self.embed_dim, 1), nn.Sigmoid())

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        # self.graph_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, l1_gate, l2_gate, e1_gate, e2_gate, idx_pairs1, idx_pairs2, encoder_padding_mask, enc_edge_padding_mask, graph_mask, graph_mask_rev, enc_edge_emb):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        # residual = x
        #
        # x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        #
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = residual + x
        # x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        # # use layer norm value, it can be modified and edge is not be normed.
        # T x B x C
        head = l1_gate(x)
        tail = l2_gate(x)

        link1 = torch.gather(head, dim=0, index=idx_pairs1)
        link2 = torch.gather(tail, dim=0, index=idx_pairs2)

        edges = e1_gate(torch.cat([link1, enc_edge_emb, link2], 2))
        edges_rev = e2_gate(torch.cat([link1, enc_edge_emb, link2], 2))

        graph_incoming, _ = self.graph_attn(query=head, key=edges, value=edges, key_padding_mask=enc_edge_padding_mask,
                                            query_padding_mask=encoder_padding_mask, graph_mask=graph_mask)
        graph_outgoing, _ = self.graph_attn(query=tail, key=edges_rev, value=edges_rev,
                                            key_padding_mask=enc_edge_padding_mask,
                                            query_padding_mask=encoder_padding_mask, graph_mask=graph_mask_rev)
        graph_incoming = F.dropout(graph_incoming, p=self.dropout, training=self.training)
        graph_outgoing = F.dropout(graph_outgoing, p=self.dropout, training=self.training)

        fusion_gate = self.fusion_gate(torch.cat([graph_incoming, graph_outgoing], 2))
        x = fusion_gate * graph_incoming + (1 - fusion_gate) * graph_outgoing

        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('transformer_stack_with_graph_copy', 'transformer_stack_with_graph_copy')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('transformer_stack_with_graph_copy', 'transformer_stack_with_graph_copy_gigaword')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.graph_attention_heads = getattr(args, 'graph_attention_heads', 4)
    args.graph_attention_dropout = getattr(args, 'graph_attention_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)

@register_model_architecture('transformer_stack_with_graph_copy', 'transformer_stack_with_graph_copy_gigaword_big')
def transformer_stack_with_graph_copy_gigaword_big2(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.graph_attention_heads = getattr(args, 'graph_attention_heads', 4)
    args.graph_attention_dropout = getattr(args, 'graph_attention_dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.graph_layers = getattr(args, 'graph_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer_stack_with_graph_copy', 'transformer_stack_with_graph_copy_gigaword_big3')
def transformer_gigaword_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.graph_attention_heads = getattr(args, 'graph_attention_heads', 8)
    args.graph_attention_dropout = getattr(args, 'graph_attention_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.2)
    # args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    # args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    base_architecture(args)

@register_model_architecture('transformer_stack_with_graph_copy', 'transformer_stack_with_graph_copy_gigaword_big4')
def transformer_stack_with_graph_copy_gigaword_big2(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.graph_attention_heads = getattr(args, 'graph_attention_heads', 4)
    args.graph_attention_dropout = getattr(args, 'graph_attention_dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 8)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 8)
    base_architecture(args)
