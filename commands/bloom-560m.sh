LOAD_PATH=/root/tmp_ckpt/bloom-560m_tp2_pp2/
CHECKPOINT_PATH=/root/tmp_ckpt/bloom-560m_tp2_pp1_lr0_tt
INIT_PATH=/root/tmp_ckpt/bloom-560m-tp2_test/global_step50/

# tokenizer of bloom is the same across multiple scales
VOCAB_FILE=data/bloom-7b1/vocab.json
MERGE_FILE=data/bloom-7b1/merges.txt
DATA_PATH=data/test-bloom7b-oscar-en-1G_text_document
TENSORBOARD_PATH=output_dir/tensorboard
TOKENIZER_NAME=bigscience/bloom-7b1

N_GPUS=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=1

NLAYERS=24
NHIDDEN=1024
NHEADS=16
SEQ_LEN=2048
VOCAB_SIZE=50257

SAVE_INTERVAL=10

TRAIN_SAMPLES=1000000

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 1200 \
    --clip-grad 1.0 \
    --finetune \
    --weight-decay 0.1 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 4 4 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --embed-layernorm \
    --position-embedding-type alibi \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME \
    --init-method-std 0.0048 \
    --fp16 \
    --seed 42 \
    --pad-vocab-size-to 250880 \
    --abort-on-unmet-fused-kernel-constraints \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --exit-interval 100000 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $LOAD_PATH \
    --init-load $INIT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    "

ZERO_STAGE=1

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": true
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

MASTER_ADDR=localhost
MASTER_PORT=6777

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "
export CMD=" \
    $LAUNCHER pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --data-impl mmap \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD
