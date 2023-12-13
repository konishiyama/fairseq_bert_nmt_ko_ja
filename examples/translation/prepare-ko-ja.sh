#!/usr/bin/env bash

# Adapted for Japanese-Korean language pair processing

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

src=ko
tgt=ja
prep=ko_ja_dataset
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from Google Drive..."
# You might need to use gdown or a similar tool to download directly from Google Drive
# gdown 'https://drive.google.com/file/d/1T57hI8AULVDJ4oe5KLUVScakQQxILga2/view?usp=sharing'
pip install gdown
gdown --id 1T57hI8AULVDJ4oe5KLUVScakQQxILga2 --output $orig/ja-ko.json
if [ $? -ne 0 ]; then
    echo "Error: Download failed. Exiting script."
    exit 1
fi
# Assuming JSON file is named 'data.json' and is already in the 'orig' directory
echo "Pre-processing JSON data..."
python -c "import json; \
           d = json.load(open('$orig/ja-ko.json')); \
           open('$tmp/data.ja', 'w').write('\n'.join([i['ja'] for i in d])); \
           open('$tmp/data.ko', 'w').write('\n'.join([i['ko'] for i in d]))"

echo "Tokenizing..."
for l in $src $tgt; do
    perl $TOKENIZER -threads 8 -l $l < $tmp/data.$l > $tmp/data.tok.$l
done

# echo "Cleaning train data..."
# perl $CLEAN -ratio 1.5 $tmp/data.tok $src $tgt $tmp/data.clean 1 175

echo "Creating train, valid, test splits..."
for l in $src $tgt; do
    awk 'NR%10==1' $tmp/data.$l > $tmp/test.$l
    awk 'NR%10==2' $tmp/data.$l > $tmp/valid.$l
    awk 'NR%10!=1 && NR%10!=2' $tmp/data.$l > $tmp/train.$l
done

TRAIN=$tmp/train.ja-ko
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

echo "Finished processing"
