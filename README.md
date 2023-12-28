#  Hot or Cold? Adaptive Temperature Sampling for Code Generation with Large Language Models

Official implementation of AdapT in AAAI 2024 [paper](https://lj2lijia.github.io/papers/AdapT_AAAI2024.pdf).

## Getting Started

### Installation

Python 3.7+ / CUDA 11+ / PyTorch 1.10+ / DeepSpeed 0.6+ are required. Install 

```bash
cd AdapT
pip install -e .
```

### Model Weights

Apply and download model weights through this [link](https://models.aminer.cn/codegeex/download/request), then you can receive by mail ```urls.txt``` that contains temporary download links. 

To downlad model weights, use [aria2](https://aria2.github.io/) to download it via the following command:

```bash
aria2c -x 16 -s 16 -j 4 --continue=true -i urls.txt 
```

Run the following command to get the full model weights:
```bash
cat codegeex_13b.tar.gz.* > codegeex_13b.tar.gz
tar xvf codegeex_13b.tar.gz
```

## Inference

Sampling results generated with AdapT sampling on HumanEval and MBPP are shown in the output files
(i.e. output/humaneval_adapt_samples.jsonl, output/mbpp_adapt_samples.jsonl)

You can follow the instructions below to get these results:

### HumanEval

To evaluate the AdapT sampling on HumanEval dataset, with 15 candidates generated, and adaptive temperatures of [0.8,0.6], run

```bash
sh scripts/inference_adapt.sh 0 inputs/HumanEval.jsonl inputs/human_eval_stop_words.json he_samples.jsonl 0.5 0.8 0.6
```

### MBPP

To evaluate the AdapT sampling on MBPP dataset, with 20 candidates generated, and adaptive temperatures of [0.8,0.3], run
```bash
sh scripts/inference_adapt.sh 0 inputs/mbpp_test.jsonl inputs/mbpp_stop_words.json mbpp_samples.jsonl 0.5 0.6 0.5
```

## Evaluation

To evaluate the generated results, install the mxeval repository:

```bash
git clone https://github.com/amazon-science/mxeval.git
pip install -e mxeval
```

### HumanEval

To get the evaluation results of HumanEval dataset, run the following command:

```bash
evaluate_functional_correctness output/humaneval_adapt_samples.jsonl --problem_file inputs/HumanEval.jsonl --k 1,5,10,15
```

### MBPP

To get the evaluation results of MBPP dataset, run the following command:

```bash
evaluate_functional_correctness output/mbpp_adapt_samples.jsonl --problem_file inputs/mbpp_test.jsonl --k 1,5,10,15
```
