# gradientlab

A lab where gradients flow and models go to prod.

# Guiding principles

- Experiment as a first-class citizen
    - full replicability: dataprep, modeling, configs, training and eval code is self-contained
- Architecture copy-paste is allowed, no preemptive optimization when doing applied AI
    - If you're reusing an exact same nn.Module *N* times, go modularize it.
    - For me N=3 means that the thing works => refactor.
- Cristalize a stable architecture or nn.Module under `neuralblocks/`
    - Avoid model overparametrization and huge configs
- HuggingFace basic compatibility
    - we don't do whitepapers, we push to prod ASAP
- Notebooks as a clean demo interface
    - do dirty & temporary stuff under `notebooks/trash`


# Install

## prereqs

uv:
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

An example is under `/experiments`, with a custom small 20M param GPT under `/modeling` with PolyReLU ffn activation, parallel attention (from PaLM paper & Moondream), absolute position embeddings and KV-cache. + Slim notebook to demo model loading and generation.


# Publish
If you want to publish ...