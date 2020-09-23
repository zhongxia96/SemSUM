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

**Citation:**
```
@inproceedings{DBLP:conf/aaai/JinW020,
  author    = {Hanqi Jin and
               Tianming Wang and
               Xiaojun Wan},
  title     = {SemSUM: Semantic Dependency Guided Neural Abstractive Summarization},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {8026--8033},
  year      = {2020},
  crossref  = {DBLP:conf/aaai/2020},
  url       = {https://aaai.org/ojs/index.php/AAAI/article/view/6312},
  timestamp = {Thu, 04 Jun 2020 13:18:48 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/JinW020.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
