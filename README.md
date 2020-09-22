# SemSUM: Semantic Dependency Guided Neural Abstractive Summarization
**Code for paper [SemSUM: Semantic Dependency Guided Neural Abstractive Summarization](https://aaai.org/ojs/index.php/AAAI/article/view/6312) by Hanqi Jin, Tianming Wang, Xiaojun Wan. This paper is accepted by AAAI'20.**

Some codes are borrowed from [fairseq](https://github.com/pytorch/fairseq).

**Requirements and Installation：**
* PyTorch version >= 1.4.0
* Python version >= 3.6

**[Download Data](https://drive.google.com/file/d/1d8ZG-V2MAEN6fAAFnGURyH4Ug5Q0pJQT/view?usp=sharing)**

**Preprocess：**
```
python preprocess_graph.py --trainpref ./gigaword_data/train --validpref ./gigaword_data/valid \
--testpref ./gigaword_data/test --source-lang src --target-lang tgt --destdir gigaword-graph \
--joined-dictionary --nwordssrc 50000 --workers 5 --edgedict ./gigaword_data/dict.edge.txt
```
```

python process_graph_copy.py --testpref ./gigaword_data/test --source-lang src --target-lang tgt \
--destdir gigaword-graph-copy  --nwordssrc 50000 --workers 5 \
--edgedict ./gigaword_data/dict.edge.txt --srcdict gigaword-graph/dict.src.txt \
--tgtdict gigaword-graph/dict.tgt.txt --dataset-impl raw
```

**Train:**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py gigaword-graph \
  -a transformer_stack_with_graph_copy_gigaword_big --optimizer adam --lr 0.0001 -s src -t tgt \
  --dropout 0.1 --max-tokens 2000 \
  --share-decoder-input-output-embed \
  --task translation_with_graph_attention_with_copy \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer-graph-gigaword --share-all-embeddings\
  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --criterion cross_entropy_copy --update-freq 2
```

**Test:**
```
CUDA_VISIBLE_DEVICES=4 python generate.py gigaword-graph-copy \
--task translation_with_graph_attention_with_copy  \
--path  checkpoints/transformer-graph-gigaword/checkpoint_best.pt \
--batch-size 128 --beam 5 --lenpen 1.2 --replace-unk --raw-text  
```

