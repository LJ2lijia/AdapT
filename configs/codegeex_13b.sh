# CodeGeeX-13B configuration

CHECKPOINT_PATH=$(realpath codegeex_13b.pt)
if [ ! -f "$CHECKPOINT_PATH" ]
then    echo cannot find checkpoint at $CHECKPOINT_PATH
        exit 1
fi

MODEL_ARGS="--num-layers 39 \
            --hidden-size 5120 \
            --num-attention-heads 40 \
            --max-position-embeddings 2048 \
            --attention-softmax-in-fp32 \
            --load "$CHECKPOINT_PATH" \
            --layernorm-epsilon 1e-5 \
            --fp16 \
            --ws-encoding-start-id 10 \
            --ws-encoding-length 10 \
            --make-vocab-size-divisible-by 52224 \
            --seq-length 2048"
