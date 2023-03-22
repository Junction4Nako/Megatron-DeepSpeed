# split bloom-560m into tp size of 2
mkdir /root/tmp_ckpt
python tools/split_checkpoints.py \
        --source_model bigscience/bloom-560m \
        --tp_size 2 \
        --save_path /root/tmp_ckpt/bloom-560m_tp2 \
        --half

# prepare the data
# bash commands/test_data.sh

# runing the model
bash commands/bloom-560m.sh