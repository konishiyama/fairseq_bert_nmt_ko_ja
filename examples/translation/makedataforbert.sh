# !/usr/bin/env bash
# lng=$1
# echo "src lng $lng"
# for sub  in train valid test
# do
#     sed -r 's/(@@ )|(@@ ?$)//g' ${sub}.${lng} > ${sub}.bert.${lng}.tok
#     ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < ${sub}.bert.${lng}.tok > ${sub}.bert.${lng}
#     rm ${sub}.bert.${lng}.tok
# done


#!/bin/bash

lng=$1
echo "src lng $lng"
directory="ko_ja_dataset" # Add the directory path

for sub in train valid test
do
    # Update file paths to include the directory
    input_file="${directory}/${sub}.${lng}"
    intermediate_file="${directory}/${sub}.bert.${lng}.tok"
    output_file="${directory}/${sub}.bert.${lng}"

    # Check if the input file exists
    if [ ! -f "$input_file" ]; then
        echo "File not found: $input_file"
        continue # Skip to the next file if the current one does not exist
    fi

    # Perform BPE removal and detokenization
    sed -r 's/(@@ )|(@@ ?$)//g' "$input_file" > "$intermediate_file"
    perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < "$intermediate_file" > "$output_file"
    rm "$intermediate_file"
done
