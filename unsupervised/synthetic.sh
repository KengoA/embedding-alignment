#!/bin/bash
SRC_LANG=z_0; # source language
TGT_LANG=z_1; # target language

DATA_ROOT=./data/;
TRAIN_MAX_SIZE=200 # the top N words included in training

python src/make_synthetic_data.py

# train the word embedding
python src/runner.py \
    --config_path src/config/config_synthetic.json \
    --src "$SRC_LANG" --tgt "$TGT_LANG" \
    --src_vec "$DATA_ROOT"synthetic/"$SRC_LANG".vec \
    --tgt_vec "$DATA_ROOT"synthetic/"$TGT_LANG".vec \
    --train_max_size "$TRAIN_MAX_SIZE" \
    --save ./exp/ \
    --train 1 \
    --F_validation "$DATA_ROOT"crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG".txt;

# evaluate the trained embeddings
python src/eval/eval_translation.py \
    exp/"$SRC_LANG"-"$TGT_LANG"/src.emb.txt exp/"$SRC_LANG"-"$TGT_LANG"/tgt.trans.emb.txt \
    -d data/crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG".txt \
    --exp_dir exp/"$SRC_LANG"-"$TGT_LANG";

python src/eval/eval_translation.py \
    exp/"$SRC_LANG"-"$TGT_LANG"/tgt.emb.txt exp/"$SRC_LANG"-"$TGT_LANG"/src.trans.emb.txt \
    -d data/crosslingual/dictionaries/"$TGT_LANG"-"$SRC_LANG".txt \
    --exp_dir exp/"$SRC_LANG"-"$TGT_LANG";