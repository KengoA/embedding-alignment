#!/bin/bash
DATA_ROOT=./data/;
TRAIN_MAX_SIZE=200 # the top N words included in training

for N_DIM in 4 10 30 50
do
    for ITER in {1..20}
    do
        python src/make_synthetic_data.py \
            --n_dim "$N_DIM" \
            --iter "$ITER" \
            --seed 42;
        
        SRC_LANG="z_0"_n_dim_"$N_DIM"_"$ITER";
        TGT_LANG="z_1"_n_dim_"$N_DIM"_"$ITER";

        # train the word embedding
        python src/runner.py \
            --config_path src/config/config_synthetic.json \
            --src "$SRC_LANG" --tgt "$TGT_LANG" \
            --src_vec "$DATA_ROOT"synthetic/"$SRC_LANG".vec \
            --tgt_vec "$DATA_ROOT"synthetic/"$TGT_LANG".vec \
            --train_max_size "$TRAIN_MAX_SIZE" \
            --save ./exp/synthetic/ \
            --train 1 \
            --F_validation "$DATA_ROOT"crosslingual/dictionaries/z_0-z_1.txt;

        # evaluate the trained embeddings
        python src/eval/eval_translation.py \
            exp/synthetic/"$SRC_LANG"-"$TGT_LANG"/src.emb.txt exp/synthetic/"$SRC_LANG"-"$TGT_LANG"/tgt.trans.emb.txt \
            -d data/crosslingual/dictionaries/z_0-z_1.txt \
            --exp_dir exp/synthetic/"$SRC_LANG"-"$TGT_LANG";

        python src/eval/eval_translation.py \
            exp/synthetic/"$SRC_LANG"-"$TGT_LANG"/tgt.emb.txt exp/synthetic/"$SRC_LANG"-"$TGT_LANG"/src.trans.emb.txt \
            -d data/crosslingual/dictionaries/z_1-z_0.txt \
            --exp_dir exp/synthetic/"$SRC_LANG"-"$TGT_LANG";

    done
done