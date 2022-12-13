from sacremoses import MosesTokenizer, MosesDetokenizer
import time
import pickle as pkl
import re
from collections import defaultdict
import nltk
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from transformers import PreTrainedTokenizerFast

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os
import sys
import argparse


def train_tokenizer(folders, lang):
    # load Pretrained tokenizer
    if os.path.exists(f"./data/wmt14_data/bpe/vocab_{lang}.json"):
        tokenizer = Tokenizer.from_file(
            f"./data/wmt14_data/bpe/vocab_{lang}.json")

    else:
        # Train tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=40000, special_tokens=["<|endoftext|>"])
        files = get_files(folders, lang)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.train(
            files, trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()
        # Save tokenizer
        tokenizer.save(f"./data/wmt14_data/bpe/vocab_{lang}.json")

    return tokenizer


def POS_tag(text, lang):
    # POS tag sentences
    pos_tagged = []
    for line in text:
        pos_tagged.append(nltk.pos_tag(line, lang=lang))
    return pos_tagged


def get_files(folders, lang):
    files = []
    for folder in folders:
        # read filenames from folder
        filenames = os.listdir(folder)
        for filename in filenames:
            if filename.endswith(f".{lang}"):
                files.append(folder+filename)
    return files


def read_txt(file):
    f = open(file, "r", encoding="utf-8")
    text = []
    data = f.read().splitlines()
    for line in data:
        text.append(line)
    return text


def save_txt(text, file):
    f = open(file, "w", encoding="utf-8")
    for line in text:
        f.write(" ".join(line)+"\n")
    f.close()
    # print("write file to:" + file)


def process_bpe(text, tokenizer):
    # recover sentences from bpe sentences
    clean = []
    for line in text:
        a = "".join(line).replace("Ġ", "")
        clean.append(a.split(" "))

    return clean


def pos_subid(pos_tag, ids, limit_dict):
    a = int(limit_dict[pos_tag])
    if a == 0:
        return pos_tag
    elif ids < a:
        return pos_tag+str(ids)
    else:
        return pos_tag+str(a)


def tokenize(folders, tokenizer, lang):
    # tokenize sentences
    files = get_files(folders, lang)

    for file in files:

        tokenized = []
        text = read_txt(file)

        for i, line in enumerate(text):
            # print(f"{i/len(text)*100:.2f}%", end="\r")
            tokenized.append(tokenizer.encode(line).tokens)
        save_txt(
            tokenized, f"data/wmt14_data/tokenized/{file.split('/')[-1]}.bpe")

    return tokenized


def get_limit_dict(limit_data):
    l_keys = []
    l_ids = []
    for line in limit_data:
        line = line.split(" ")
        l_keys.append(line[0])
        l_ids.append(line[1])
    return dict(zip(l_keys, l_ids))


def getpos_bpe(data1, data2, limit_dict):
    """
    data1 = list of bpe sentences
    data2 = list of raw sentences
    """
    pos_test = []
    time1 = time.time()
    for i in range(len(data1)):
        if i % 100000 == 1:
            print(i)
            print(time.time()-time1)
            print("#"*66)
            time1 = time.time()
        pos = []
        line = data1[i].split(" ")
        pos_line = nltk.pos_tag(data2[i])
        p_line = []
        for w, p in pos_line:
            if w == p or w in [":", "?", "-", "...", ";", "--", "!"] or p in ["(", ")", "``"]:
                p_line.append("PCT")
            else:
                # combining "(",")"and"``" are named as PCT,
                # "''"and"$" are named as SYM$, WP$ is combined to WP
                if p in ["$", "''", "SYM"]:
                    p_line.append("SYM$")
                elif p == "WP$":
                    p_line.append("WP")
                else:
                    p_line.append(p)
        j = 0
        n = 0

        while j < len(line):
            if j == len(line)-1:
                pos.append(p_line[n])
                break
            else:
                if "##" not in line[j+1] or "Ġ" in line[j+1]:
                    pos.append(p_line[n])
                    j += 1
                else:
                    k = 1
                    while k:
                        if j+k == len(line):
                            pos.append(pos_subid(p_line[n], k, limit_dict))
                            break
                        pos.append(pos_subid(p_line[n], k, limit_dict))
                        if "##" not in line[j+k]:
                            #                         pos.append(pos_subid(p_line[n],k+1))
                            break
                        k += 1
                    j = j+k
                n += 1

        assert len(pos) == len(line)
        pos_test.append(pos)
    return pos_test
