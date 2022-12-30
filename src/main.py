import argparse
import os
import sys
from preprocess import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train",
                        default="/data/wmt14_data/train/")
    parser.add_argument("--valid",
                        default="/data/wmt14_data/valid/")
    parser.add_argument("--test",
                        default="/data/wmt14_data/test/")
    parser.add_argument("--bpe", default="/data/wmt14_data/bpe/")
    parser.add_argument("--limit", default="/data/pos_limit100.txt")
    parser.add_argument("--lang", default="en")
    parser.add_argument("-t", "--tokenize",  default=False,
                        action='store_true')
    parser.add_argument("-p", "--pos", default=False, action='store_true')

    args = parser.parse_args()
    folders = [args.test, args.valid, args.train]
    lang = args.lang
    bpe_folder = args.bpe
    limit_data = read_txt(args.limit)

    tokenizer = train_tokenizer(folders, lang)
    if args.tokenize:
        print("Tokenizing")
        tokenize(folders, tokenizer, lang)

    if args.pos:
        limit_dict = get_limit_dict(limit_data)
        files = get_bpe_files(bpe_folder, lang)

        for file in files:
            print(f"POS tagging {file}")
            data1 = read_txt(file)
            print("preprocess bpe sentences")
            data2 = process_bpe(data1, tokenizer)
            print("get pos tags")
            pos = getpos_bpe(data1, data2, limit_dict)
            save_txt(pos, file+".pos")
