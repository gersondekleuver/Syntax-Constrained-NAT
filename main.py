import argparse
import os
import sys
from preprocess import train_tokenizer, get_files, tokenize

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true",
                        default="data/wmt14_data/train/")
    parser.add_argument("--valid", action="store_true",
                        default="data/wmt14_data/valid/")
    parser.add_argument("--test", action="store_true",
                        default="data/wmt14_data/test/")
    parser.add_argument("--lang", action="store_true", default="en")

    args = parser.parse_args()

    folders = [args.train, args.valid, args.test]
    lang = args.lang

    tokenizer = train_tokenizer(folders, lang)
    tokenize(folders, tokenizer, lang)
