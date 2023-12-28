
import os
import copy
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from json import loads, dumps
import regex as re
from pathlib import Path

from codegeex.torch.inference_ori import get_token_stream
from codegeex.torch import CodeGeeXModel
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.quantization import quantize


def model_provider(args):
    """Build the model."""

    model = CodeGeeXModel(
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.padded_vocab_size,
        args.max_position_embeddings
    )

    return model


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="code generation")
    group.add_argument(
        "--num-layers",
        type=int,
        default=39,
    )
    group.add_argument(
        "--hidden-size",
        type=int,
        default=5120,
    )
    group.add_argument(
        "--num-attention-heads",
        type=int,
        default=40,
    )
    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=52224,
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--sample-n",
        type=int,
        default=1,
        help="Number of samples to generate per prompt.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--prompt-file",
        type=str,
        required=True,
    )
    group.add_argument(
        "--output-file",
        type=str,
        required=True,
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizer",
    )
    group.add_argument(
        "--load",
        type=str,
    )
    group.add_argument(
        "--state-dict-path",
        type=str,
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )
    group.add_argument(
        '--out_seq_length',
        type=int,
        default=2048,
        help='Size of the output generated text.'
    )
    group.add_argument(
        '--stop-words',
        type=str,
        default='def,class',
        help='Stop words to stop the generation.'
    )
    group.add_argument(
        '--stop-words-json',
        type=Path,
        default='',
        help='Stop words to stop the generation.'
    )
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()

    temperature = args.temperature
    greedy = temperature == 0.0
    if greedy:
        temperature = 1.0
        args.sample_n = 1


    print("Loading tokenizer ...")
    tokenizer = CodeGeeXTokenizer(
        tokenizer_path=args.tokenizer_path,
        mode="codegeex-13b")

    if args.stop_words_json.exists():
      STOP_WORDS = loads(args.stop_words_json.read_text())
    else:
      STOP_WORDS = [word.strip() for word in args.stop_words.split(",")]

    def get_stop_ids(stop_words):
        stop_ids = []
        for word in stop_words:
            id = tokenizer.encode_code(word)
            assert len(id) == 1
            stop_ids.append(id[0])
        return stop_ids

    # STOP_IDS = get_stop_ids(STOP_WORDS)

    def get_stop_regex(stop_words):
        return re.compile(rf"(.*?)(?:{'|'.join(re.escape(word) for word in stop_words)}).*", re.DOTALL)

    STOP_REGEX = get_stop_regex(STOP_WORDS)

    print("Loading state dict ...")
    state_dict = torch.load(args.load, map_location="cpu")
    print("args.load", args.load)
    state_dict = state_dict["module"]

    print("Building CodeGeeX model ...")
    model = model_provider(args)
    model.load_state_dict(state_dict)
    model.eval()
    model.half()
    if args.quantize:
        model = quantize(model, weight_bit_width=8, backend="torch")
    model.cuda()

    with open(args.prompt_file, "r") as f:
        prompts = [loads(line) for line in f.readlines()]

    out_seq_length = args.out_seq_length
    seq_length = args.max_position_embeddings

    bar = tqdm(total=len(prompts) * args.sample_n)
    with open(args.output_file, "w") as f:
        for json in prompts:
            json['completion'] = []
            prompt = json['prompt']
            if 'example' in json:
                prompt = json['example'] + '\n' + prompt
            tokens = tokenizer.encode_code(prompt)
            n_token_prompt = len(tokens)
            for _ in range(args.sample_n):
                token_stream = get_token_stream(
                    model,
                    tokenizer,
                    seq_length,
                    out_seq_length,
                    [copy.deepcopy(tokens)],
                    micro_batch_size=1,
                    topk=args.top_k,
                    topp=args.top_p,
                    temperature=temperature,
                    greedy=greedy,
                )
                for generated_tokens, _ in token_stream:
                    if generated_tokens[0].cpu().numpy()[-1] == tokenizer.eos_token_id:
                        break
                    elif len(generated_tokens[0]) >= out_seq_length + n_token_prompt:
                        break
                    elif (
                        len(generated_tokens[0]) >= n_token_prompt + 2
                        # and any(generated_tokens[0].cpu().numpy()[-2] for id in in STOP_IDS)
                        and STOP_REGEX.match("".join(tokenizer.decode_code(generated_tokens[0].cpu().numpy().tolist()[n_token_prompt:])))
                    ):
                        break
                    else:
                        pass
                else:
                    raise RuntimeError("Failed to generate code.")
                generated_tokens_ = generated_tokens[0].cpu().numpy().tolist()
                # if generated_tokens_[-1] in STOP_IDS:
                #     generated_tokens_ = generated_tokens_[:-1]
                if generated_tokens_[-1] == tokenizer.eos_token_id:
                    generated_tokens_ = generated_tokens_[:-1]
                generated_code = tokenizer.decode_code(generated_tokens_[n_token_prompt:])
                generated_code = "".join(generated_code)
                if STOP_REGEX.match(generated_code):
                    generated_code = STOP_REGEX.match(generated_code).group(1)
                f.write(dumps(dict(task_id=json['task_id'], prompt=prompt, completion=generated_code, language='python')) + '\n')
                bar.update(1)


    bar.close()


if __name__ == "__main__":
    main()
