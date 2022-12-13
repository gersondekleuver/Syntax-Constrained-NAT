# Syntax-Constrained-NAT

## Preprocess

```
python fairseq/fairseq_cli/preprocess.py --workers 4 --tokenizer "moses" --bpe "byte_bpe" --source-lang "de" --target-lang "en" --destdir "data/wmt14_data/bpe" --trainpref data/wmt14_data/train/europarl-v7.de-en --validpref data/wmt14_data/valid/newstest2013 --testpref data/wmt14_data/test/newstest2014
```

!python train.py /content/drive/MyDrive/mini-project-A/ro-en-fairseq-bin \
 --save-dir /content/drive/MyDrive/mini-project-A/model/chrf_mrt/ \
 --task translation_lev \
 --criterion mrt_loss \
 --metric chrf \
 --arch cmlm_transformer \
 --noise random_mask \
 --share-all-embeddings \
 --optimizer adam --adam-betas '(0.9,0.98)' --reset_optimizer \
 --lr 1e-6 --lr-scheduler inverse_sqrt \
 --dropout 0.3 --weight-decay 0.01 \
 --decoder-learned-pos \
 --encoder-learned-pos \
 --apply-bert-init \
 --log-format 'simple' --log-interval 100 \
 --fixed-validation-seed 7 \
 --max-tokens 8000 \
 --save-interval-updates 200\
 --max-update 1000
