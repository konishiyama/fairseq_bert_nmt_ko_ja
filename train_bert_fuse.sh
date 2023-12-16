#!/usr/bin/env bash
nvidia-smi

cd /yourpath/bertnmt
python3 -c "import torch; print(torch.__version__)"

src=ko
tgt=ja
bedropout=0.5
ARCH=transformer_s2_iwslt_de_en
# トレーニングデータのパス
DATAPATH=data-bin/ko_ja_preprocessed
# Bertfuseで訓練したモデルの置き場
SAVEDIR=checkpoints/bert_fuse_${src}_${tgt}_${bedropout}  
mkdir -p $SAVEDIR
# もしcheckpoint_nmt.ptという名前のNMTモデルがSAVEDIRに存在しない場合、指定のパスからコピーしてくる（your_pretrained_nmt_modelにVanillaNMTモデル（checkpoint_best.ptかな）を指定、checkpoint_nmt.ptという名前でSAVEDIRに保存する。）
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp checkpoints/vanilla_nmt_ko_ja/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
fi
# checkpoint_last.ptが存在しない場合はwarmup変数を指定する。（warmup関連変数があるとVanillaモデルを基準に学習を始める設定が行われる（checkpointの設定）。）
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi


python train_bert_fuse.py $DATAPATH \
--save-dir $SAVEDIR \
-arch $ARCH --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--warmup-init-lr '1e-07' --min-lr '1e-09' \
--dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4096 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric
-s $src -t $tgt \
--max-update 150000 \
--share-all-embeddings $warmup \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout | tee -a $SAVEDIR/training.log


# 下記は通常のfairseqのトレーニングの場合。下記にリストするパラメータを、bertnmtの例に追加している。
# --share-decoder-input-output-embed \
# --clip-norm 0.0 \
# --max-tokens 4096 \
# --eval-bleu \
# --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses \
# --eval-bleu-remove-bpe \
# --eval-bleu-print-samples \
# --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# 通常のfairseqのトレーニングのコマンドライン
# CUDA_VISIBLE_DEVICES=0 fairseq-train \
#     data-bin/iwslt14.tokenized.de-en \
#     --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric