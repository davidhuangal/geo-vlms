# Reproducibility

A run is meant to be byte-for-byte repeatable on the same machine and software.

What makes it so:
- Greedy decoding on both backends.
- Dataset sampling seeded by `seed`, with sorted file lists.
- Model revision, library versions, and server build recorded in `.meta.json`.
- `dataset.sha256` fingerprints the exact questions and labels.

What breaks it:
- A different torch, transformers, or llama.cpp build.
- A different device or dtype. MPS, CUDA, and CPU can disagree in low bits.
- A different GGUF quantization or llama-server flags.
- A moved HF model revision. Compare `backend.commit_hash`.

## Check

```bash
for run in a b; do
  uv run geo-vlms \
    backend=huggingface \
    model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    dataset.num_pos=3 \
    dataset.no_neg=true \
    out=/tmp/repro_$run.jsonl
done
diff /tmp/repro_a.jsonl /tmp/repro_b.jsonl
```

Records should be identical.
Sidecars should differ only in `command`, `args.out`, and `started_at`.
Redo this after bumping torch or transformers.
The same check works against a llama-server.
