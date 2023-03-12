#prepare the data for test
mkdir data
mkdir data/test
cd data/test
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d ./oscar-1GB.jsonl.xz

cd ../../
# get the vocabulary file
python tools/get_tokenizer_vocab.py \
    --tokenizer_name bigscience/bloom-7b1 \
    --save_path ./data/bloom-7b1/

# preprocessing the data
python tools/preprocess_data.py \
    --input ./data/test/oscar-1GB.jsonl \
    --output-prefix data/test-bloom7b-oscar-en-1G \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path bigscience/bloom-7b1 \
    --append-eod \
    --workers 8