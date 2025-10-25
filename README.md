# gradientlab

A lab where gradients flow and models go to prod.

This repo is an attempt to have a tidy place for my own small scale pytorch-based deep learning experiments.

# Guiding principles

- Experiment as a first-class citizen
    - full replicability: dataprep, modeling, configs, training and eval code is self-contained
- Architecture copy-paste is allowed, no preemptive optimization when doing applied AI
    - Still, we're not savages: If you're reusing an exact same nn.Module *N* times, go modularize it.
    - For me N=3 means that the thing works => refactor.
- Cristalize a stable architecture or nn.Module under `neuralblocks/`
    - Avoid model overparametrization and huge configs
- HuggingFace basic compatibility
    - we don't do whitepapers, we push to prod ASAP
- Notebooks as a clean demo interface
    - do dirty & temporary stuff under `notebooks/trash`
- ...

If you want to fork the repo or install it, keep reading.

# Install

## prereqs

- A linux box with CUDA or apple silicon.
    - Rocm may work as well, not tested
- uv:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## As your own personal lab -> Fork this repo and clone it

```
git clone https://github.com/<your-github-user>/gradientlab.git
cd gradientlab/
uv sync
```

## As a library

```
uv add gradientlab
```

# Experiments

An example is under `/experiments`, a custom 22-layers, yet only 20M param GPT, which you can find under `/modeling`:
- PolyReLU ffn activation (works better than SwiGLU)
- parallel attention (from PaLM paper & Moondream)
- squeeze-and-excite narrow transformer backbone (an idea of mine for small lang models, prefering depth over width, inspired by computer vision)
- sigmoid gating post sdpa ([paper by Qwen team](https://arxiv.org/abs/2505.06708))
- attn values heads expansion
- absolute position embeddings (I know)
- KV-cache support
- embed_dim != hidden_dim
- Trained on 3B italian tokens from fineweb2 in ~8 hours on a RTXA4000.
    - byte_level_tokenizer, couldn't use qwen3 tokenizer due to memory constraints (gpu poor) and weird torch.compile errors
- Slim notebook to demo model loading and generation.
- single-GPU trainer with trackio to track metrics


Each experiment entrypoint is located in `__main__.py`
So you can run an experiment like this:

```
uv run -m gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita
```

The `modeling/` folder under an experiment will contain all the modules your model is made of. Some notes:
- factory.py -> model factory, is where you will construct the models with specific parameters
- model_cfg.py -> model config class
- model.py -> your high-level model class, extending some hf class or mixins

Feel free to adapt the repo as you wish and share your learnings in the discussion section.

# Publish
If you want to publish your own `gradientlab-*` project as library, just create a PyPI token and follow the official uv [guide](https://docs.astral.sh/uv/guides/package/).

Generally as simple as:

```
uv build
UV_PUBLISH_TOKEN=pypi-your-token uv publish
```