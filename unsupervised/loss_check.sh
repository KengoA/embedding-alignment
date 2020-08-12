#!/bin/bash
DATA_ROOT=./data/;
TRAIN_MAX_SIZE=429; # the top N words included in training
N_DIM=30;

python src/make_dictionaries.py

#!/bin/bash
MODALITY=text;
EMB_TYPE=nn;
IDX_SRC=0;
IDX_TGT=1;

python src/make_real_data.py \
    --n_dim "$N_DIM" \
    --n_concept "$TRAIN_MAX_SIZE" \
    --emb_type "$EMB_TYPE" \
    --modality_src "$MODALITY" \
    --modality_tgt "$MODALITY" \
    --idx_src "$IDX_SRC" \
    --idx_tgt "$IDX_TGT";

# train the word embedding
python src/runner.py \
    --config_path src/config/config_loss_check.json \
    --src "$EMB_TYPE"_"$MODALITY"_"$IDX_SRC" --tgt "$EMB_TYPE"_"$MODALITY"_"$IDX_TGT" \
    --src_vec "$DATA_ROOT"real/"$EMB_TYPE"_"$MODALITY"_"$IDX_SRC".vec \
    --tgt_vec "$DATA_ROOT"real/"$EMB_TYPE"_"$MODALITY"_"$IDX_TGT".vec \
    --train_max_size "$TRAIN_MAX_SIZE" \
    --save ./exp/loss_check/ \
    --train 1 \
    --F_validation "$DATA_ROOT"crosslingual/dictionaries/src-tgt.txt;

# evaluate the trained embeddings
EXP_NAME="$EMB_TYPE"_"$MODALITY"_"$IDX_SRC"-"$EMB_TYPE"_"$MODALITY"_"$IDX_TGT";

python src/eval/eval_translation.py \
    exp/loss_check/"$EXP_NAME"/src.emb.txt exp/loss_check/"$EXP_NAME"/tgt.trans.emb.txt \
    -d data/crosslingual/dictionaries/src-tgt.txt \
    --exp_dir exp/loss_check/"$EXP_NAME";

python src/eval/eval_translation.py \
    exp/loss_check/"$EXP_NAME"/tgt.emb.txt exp/loss_check/"$EXP_NAME"/src.trans.emb.txt \
    -d data/crosslingual/dictionaries/src-tgt.txt \
    --exp_dir exp/loss_check/"$EXP_NAME";


