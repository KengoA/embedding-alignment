SRC_LANG=text; # source language
TGT_LANG=image; # target language

DATA_ROOT=./data/;

TRAIN_MAX_SIZE=429; # the top N words included in training

# export CUDA_VISIBLE_DEVICES=3;

python src/make_real_data.py

# train the word embedding
python src/runner.py \
    --config_path src/config/config.json \
    --src "$SRC_LANG" --tgt "$TGT_LANG" \
    --src_vec "$DATA_ROOT"real/"$SRC_LANG".vec \
    --tgt_vec "$DATA_ROOT"real/"$TGT_LANG".vec \
    --train_max_size "$TRAIN_MAX_SIZE" \
    --save ./exp/ \
    --train 1 \
    --F_validation "$DATA_ROOT"crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG".txt;

# evaluate the trained embeddings
python src/eval/eval_translation.py \
    src/exp/"$SRC_LANG"-"$TGT_LANG"/src.emb.txt src/exp/"$SRC_LANG"-"$TGT_LANG"/tgt.trans.emb.txt \
    -d data/crosslingual/dictionaries/"$SRC_LANG"-"$TGT_LANG".txt \

python src/eval/eval_translation.py \
    src/exp/"$SRC_LANG"-"$TGT_LANG"/tgt.emb.txt src/exp/"$SRC_LANG"-"$TGT_LANG"/src.trans.emb.txt \
    -d data/crosslingual/dictionaries/"$TGT_LANG"-"$SRC_LANG".txt \
