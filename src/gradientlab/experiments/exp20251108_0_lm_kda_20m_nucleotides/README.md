decoder-only transformer, 256 hidden dim, 22-layers, 20M params:
- PolyReLU ffn activation (works better than SwiGLU)
- kimi linear attention KDA
- parallel attention formulation (from PaLM paper & Moondream)
- absolute position embeddings (I know)
- KV-cache support
- embed_dim != hidden_dim
- Trained on 3B italian tokens from fineweb2 in ~8 hours on a RTXA4000.
    - byte_level_tokenizer, couldn't use qwen3 tokenizer due to memory constraints (gpu poor) and weird torch.compile errors
- Slim notebook to demo model loading and generation.
- single-GPU trainer with trackio to track metrics

cuda / ROCm only:

```uv add gradientlab[fla]```

# Results

25 epochs, Train loss 0.2, Val loss ?