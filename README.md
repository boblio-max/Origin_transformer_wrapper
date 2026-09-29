# Origin_transformer_wrapper

so I had this idea: what if you could fine-tune Hugging Face transformer models *from inside the Origin language*? this repo is the bridge that makes that work — thin Origin wrappers that call down into real PyTorch training code.

## how it actually works

two layers:
- `transformer.or` — the Origin side: `init_transformer`, `tf_config`, `load_data`, `train`, `generate`, etc. reads like English, runs like Python.
- `transformer.py` — the Python side: an `OriginDataset` + Hugging Face `Trainer` setup that does the actual gradient updates.
- `lora.py` — PEFT `LoraConfig` so you're training low-rank adapters instead of full weights (way cheaper, fits on one GPU).

```bash
pip install -r requirements.txt  # torch, transformers, peft
origin transformer.or
```

## stack

Origin language + Python, Hugging Face `transformers`, PEFT/LoRA, PyTorch. this is the same bridge pattern as `lib-collection`'s transformer backend, just split into its own repo.
