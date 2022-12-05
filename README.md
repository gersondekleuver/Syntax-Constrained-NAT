# Syntax-Constrained-NAT

## Preprocess

```
python fairseq/fairseq_cli/preprocess.py --workers 4 --tokenizer "moses" --bpe "byte_bpe" --source-lang "de" --target-lang "en" --destdir "data/wmt14_data/bpe" --trainpref data/wmt14_data/train/europarl-v7.de-en --validpref data/wmt14_data/valid/newstest2013 --testpref data/wmt14_data/test/newstest2014
```
