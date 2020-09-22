# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_mask(key, left_pad):
        return data_utils.collate_masks(
            [s[key] for s in samples],
            1, left_pad,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    enc_edge_ids = merge('enc_edge_ids', left_pad=left_pad_target)
    enc_edge_links1 = merge('enc_edge_links1', left_pad=left_pad_target)
    enc_edge_links2 = merge('enc_edge_links2', left_pad=left_pad_target)
    graph_mask = merge_mask('graph_mask', left_pad=left_pad_target)
    graph_mask_rev = merge_mask('graph_mask_rev', left_pad=left_pad_target)

    # sort by descending source length
    enc_edge_ids = enc_edge_ids.index_select(0, sort_order)
    enc_edge_links1 = enc_edge_links1.index_select(0, sort_order)
    enc_edge_links2 = enc_edge_links2.index_select(0, sort_order)
    graph_mask = graph_mask.index_select(0, sort_order)
    graph_mask_rev = graph_mask_rev.index_select(0, sort_order)




    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'enc_edge_ids': enc_edge_ids,
            'enc_edge_links1': enc_edge_links1,
            'enc_edge_links2': enc_edge_links2,
            'graph_mask': graph_mask,
            'graph_mask_rev': graph_mask_rev,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairWithGraphDataset2(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        edge (torch.utils.data.Dataset, optional): edge dataset to wrap
        edge_sizes (List[int], optional): edge type counts
        edge_dict (~fairseq.data.Dictionary, optional): edge type vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        edge=None, edge_sizes=None, edge_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024, max_edge_positions=-1,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.edge = edge
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.edge_sizes = np.array(edge_sizes) if edge_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.edge_dict = edge_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.max_edge_positions = max_edge_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        edge_item = self.edge[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        edge_item = torch.unbind(edge_item.reshape(-1, 3), 0)
        src_len = len(src_item)
        enc_edge_links1 = torch.zeros(len(edge_item)+len(src_item)-1)
        enc_edge_links2 = torch.zeros(len(edge_item)+len(src_item)-1)
        enc_edge_ids = torch.zeros(len(edge_item)+len(src_item)-1)

        # initialize mask with one, one will be mask when graph attention
        graph_mask = torch.ones(len(src_item), len(edge_item)+len(src_item)-1)
        graph_mask_rev = torch.ones(len(src_item), len(edge_item)+len(src_item)-1)

        count = 0
        for i, (head, relation, tail) in enumerate(edge_item):
            # head is head node idx, relation is relation idx between head and tail, tail is tail node idx
            if int(head) == -1:
                continue
            enc_edge_links1[i] = int(head)
            enc_edge_ids[i] = int(relation)
            enc_edge_links2[i] = int(tail)
            graph_mask[int(head)][i] = 0
            graph_mask_rev[int(tail)][i] = 0
            count += 1

        for i in range(src_len-1):
            # add adjacent edge (i, 'adjacent', i+1)
            enc_edge_links1[i+count] = i
            enc_edge_ids[i+count] = self.edge_dict.index('adjacent')
            enc_edge_links2[i+count] = i+1
            graph_mask[i][i+count] = 0
            graph_mask_rev[i+1][i+count] = 0

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'enc_edge_ids': enc_edge_ids.type(torch.LongTensor),
            'enc_edge_links1': enc_edge_links1.type(torch.LongTensor),
            'enc_edge_links2': enc_edge_links2.type(torch.LongTensor),
            'graph_mask': graph_mask.type(torch.ByteTensor),
            'graph_mask_rev': graph_mask_rev.type(torch.ByteTensor),
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        self.edge.prefetch(indices)
