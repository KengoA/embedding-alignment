#!/bin/bash
DATA_ROOT=./data/;
TRAIN_MAX_SIZE=429; # the top N words included in training
N_DIM=30;

python src/make_dictionaries.py

EMB_TYPE="normal";
IDX_SRC=3;
IDX_TGT=3;

python src/make_real_data.py \
    --n_dim "$N_DIM" \
    --n_concept "$TRAIN_MAX_SIZE" \
    --emb_type "$EMB_TYPE" \
    --modality_src "text" \
    --modality_tgt "image" \
    --idx_src "$IDX_SRC" \
    --idx_tgt "$IDX_TGT";

# train the word embedding
python src/runner.py \
    --config_path src/config/config.json \
    --src "$EMB_TYPE"_"text"_"$IDX_SRC" --tgt "$EMB_TYPE"_"image"_"$IDX_TGT" \
    --src_vec "$DATA_ROOT"real/"$EMB_TYPE"_"text"_"$IDX_SRC".vec \
    --tgt_vec "$DATA_ROOT"real/"$EMB_TYPE"_"image"_"$IDX_TGT".vec \
    --train_max_size "$TRAIN_MAX_SIZE" \
    --save ./exp/ \
    --train 1 \
    --F_validation "$DATA_ROOT"crosslingual/dictionaries/src-tgt.txt;

# evaluate the trained embeddings
EXP_NAME="$EMB_TYPE"_"text"_"$IDX_SRC"-"$EMB_TYPE"_"image"_"$IDX_TGT";

python src/eval/eval_translation.py \
    exp/"$EXP_NAME"/src.emb.txt exp/"$EXP_NAME"/tgt.trans.emb.txt \
    -d data/crosslingual/dictionaries/src-tgt.txt \
    --exp_dir exp/"$EXP_NAME";

python src/eval/eval_translation.py \
    exp/"$EXP_NAME"/tgt.emb.txt exp/"$EXP_NAME"/src.trans.emb.txt \
    -d data/crosslingual/dictionaries/src-tgt.txt \
    --exp_dir exp/"$EXP_NAME";



