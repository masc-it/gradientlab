decoder-only transformer, 256 hidden dim, 22-layers, 20M params:
- PolyReLU ffn activation (works better than SwiGLU)
- kimi linear attention KDA
- parallel attention formulation (from PaLM paper & Moondream)
- absolute position embeddings (I know, not SOTA)
- KV-cache support
- embed_dim != hidden_dim
- Slim notebook to demo model loading and generation.
- single-GPU trainer with trackio to track metrics

cuda / ROCm only:

```
uv sync
uv add flash-linear-attention
```

run experiment:

```
uv run src/gradientlab/experiments/exp20251109_0_lm_kda_20m_nucleotides_std/main.py
```

# Results

...