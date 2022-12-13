import argparse
import os
import sys
from preprocess import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train",
                        default="data/wmt14_data/train/")
    parser.add_argument("--valid",
                        default="data/wmt14_data/valid/")
    parser.add_argument("--test",
                        default="data/wmt14_data/test/")
    parser.add_argument("--lang", default="en")
    parser.add_argument("-t", "--tokenize",  default=False)
    parser.add_argument("-p", "--pos", default=False)
    parser.add_argument("--limit", default="data/pos_limit100.txt")

    args = parser.parse_args()
    folders = [args.train, args.valid, args.test]
    lang = args.lang
    limit_data = read_txt(args.limit)

    tokenizer = train_tokenizer(folders, lang)
    if args.tokenize:

        print("Tokenizing", args.tokenize)
        tokenize(folders, tokenizer, lang)

    if args.pos:
        print("POS tagging")
        limit_dict = get_limit_dict(limit_data)
        test_data1 = read_txt(
            "data/wmt14_data/tokenized/newstest2014.en.bpe")  # bpe
        print("preprocess bpe sentences")
        test_data2 = process_bpe(test_data1, tokenizer)  # without bpe
        print("get pos tags")
        test_pos = getpos_bpe(test_data1, test_data2, limit_dict)
        save_txt(test_pos, "data/wmt14_data/pos/newstest2014.en.pos")
