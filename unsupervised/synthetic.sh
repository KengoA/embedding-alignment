SRC_LANG=z_0; # source language
TGT_LANG=z_1; # target language

DATA_ROOT=./data/;

# VAL_SPLIT=0-5000 # validation data. Note that this is not used for any model selection
# TRN_SPLIT=0-5000
TRAIN_MAX_SIZE=200 # the top N words included in training
# TRANS_MAX_SIZE= # the top M words include for testing

# export CUDA_VISIBLE_DEVICES=3;

python src/make_synthetic_data.py

# train the word embedding
python src/runner.py \
    --config_path src/config/config.json \
    --src "$SRC_LANG" --tgt "$TGT_LANG" \
    --src_vec "$DATA_ROOT"synthetic/"$SRC_LANG".vec \
    --tgt_vec "$DATA_ROOT"synthetic/"$TGT_LANG".vec \
    --train_max_size "$TRAIN_MAX_SIZE" \
    --save ./exp/ \
    --train 1 \
    --F_validation "$DATA_ROOT"crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG".txt;

# evaluate the trained embeddings
python src/eval/eval_translation.py \
    ./src/exp/"$SRC_LANG"-"$TGT_LANG"/src.emb.txt ./src/exp/"$SRC_LANG"-"$TGT_LANG"/tgt.trans.emb.txt \
    -d data/crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG".txt \

python src/eval/eval_translation.py \
    ./src/exp/"$SRC_LANG"-"$TGT_LANG"/tgt.emb.txt ./src/exp/"$SRC_LANG"-"$TGT_LANG"/src.trans.emb.txt \
    -d data/crosslingual/dictionaries/"$TGT_LANG"-"$SRC_LANG".txt \
