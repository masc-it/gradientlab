# GOAL

Train a decoder-only transformer for some grokking experiments. 

## Model architecture

2 layers, 256 hidden, SwiGLUFeedForward from neuralblocks module with causal masked attention (you can use FlashMHA which is under modeling/model.py in cwd, with use_alibi=True so no pos embedding is needed), pre-norm, use RMSNorm, untied embedding and lm head.

Tokenizer: use byte_tokenizer() function

## Datasets

Under datasets, you'll create the different synthetic data to be consumed via torch.Dataset.

Let's just start with one called "DateToISO", that given a date formatted arbitrarily (think of different formats, including ISO), it converts back to ISO. The dataset generates dates from 1900-01-01 to 2100-01-01.

A collate class will left-pad the tokens and return a batch as dictionary.

## Trainer 

Take inspiration from src/gradientlab/experiments/exp20251227_imagetextzip/trainer.py and call it in a main.py (src/gradientlab/experiments/exp20251227_imagetextzip/main.py)

Create an exp_config.py (src/gradientlab/experiments/exp20251227_imagetextzip/exp_config.py) with all exp params and dataset name.

## LR scheduler

for the lr scheduler, you need to implement something different:

- warmup stage from min_lr to max_lr
- cosine decay to min_lr
- as soon as training accuracy >= 99%, ramp up lr following a cosine schedule for an epoch
- stays constant for the remaining steps

## Training auto stopping criteria

- Stop when eval accuracy ~100%

## Defaults

- min_lr: 1e-4
- max_lr: 1e-3
- weight decay: 1.0
- dropout only for residuals in transformer block: 0.1
- train_ratio: 0.05
