# This script is used to test the inference of CodeGeeX.

GPU=$1
PROMPT_FILE=$2
STOP_WORDS_JSON=$3
OUTPUT_FILE=$4
TEMP=$5
t_1=$6
t_2=$7

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"

if [ -z "$GPU" ]; then
  echo "Please specify GPU ID."
  exit 1
fi
if [ -z "$PROMPT_FILE" ]; then
  echo "Please specify prompt file."
  exit 1
fi
PROMPT_FILE=$(realpath "$PROMPT_FILE")
if [ -z "$OUTPUT_FILE" ]; then
  echo "Please specify output file."
  exit 1
fi
OUTPUT_FILE=$(realpath "$OUTPUT_FILE")
if [ -z "$TEMP" ]; then
  echo "Please specify temperature"
  exit 1
fi
if [ -z "$STOP_WORDS_JSON" ]; then
  echo "Please specify stop words json"
  exit 1
fi
STOP_WORDS_JSON=$(realpath "$STOP_WORDS_JSON")

export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=$GPU


# remove --greedy if using sampling
CMD="python -u $MAIN_DIR/tests/inference_adapt.py \
        --prompt-file $PROMPT_FILE \
        --tokenizer-path $TOKENIZER_PATH \
        --output-file $OUTPUT_FILE \
        --out-seq-length 500 \
        --temperature $TEMP \
        --t_1 $t_1 \
        --t_2 $t_2 \
        --top-p 0.95 \
        --top-k 0 \
        --sample-n 15 \
        --stop-words-json $STOP_WORDS_JSON \
        $MODEL_ARGS"

echo "$CMD"
eval "$CMD"
