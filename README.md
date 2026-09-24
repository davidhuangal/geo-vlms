# GEO-VLMs

Evaluates vision-language models on remote-sensing imagery.
It asks questions like "Are there any ships in this image?" and "How many ships are there in this image?", then scores the answers against ground truth.
Every run records its config, model revision, and software versions, so results can be reproduced.

## Quickstart

1. Install:

   ```bash
   uv sync
   ```

2. Download [NWPU VHR-10](https://gcheng-nwpu.github.io/) and unpack it under `data/vhr10` ([layout](docs/datasets.md#nwpu-vhr-10)).

3. Ask SmolVLM2 to count objects in 7 images:

   ```bash
   uv run geo-vlms \
     backend=huggingface \
     model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
     task=counting \
     dataset.num_pos=5 \
     dataset.num_neg=2
   ```

4. Score the answers:

   ```bash
   uv run scripts/analyze.py \
     --task counting \
     --records results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.jsonl
   ```

   ```text
         valid  exact_match  absolute_error  signed_error  relative_error  within_1
   mean    1.0     0.671429        0.714286           0.6        0.166667  0.842857
   ```

## What's supported

| | Supported |
|---|---|
| [Tasks](docs/tasks.md) | existence, counting |
| [Datasets](docs/datasets.md) | NWPU VHR-10, DIOR, or [your own](docs/extending.md) |
| [Backends](docs/backends.md) | HuggingFace transformers, llama-server, or [your own](docs/extending.md) |

## Learn more

1. [First eval](docs/tutorials/01-first-eval.md)
2. [llama-server](docs/tutorials/02-llama-server.md)
3. [Compare models](docs/tutorials/03-compare-models.md)
4. [Plug in a dataset and a backend](docs/tutorials/04-plug-in.md)

Reference: [docs/](docs/README.md).

## Contributing

```bash
uv sync
git config core.hooksPath .githooks
uv run pytest
uv run ruff check .
uv run ruff format .
```

The pre-commit hook runs ruff, the pre-push hook also runs pytest, and CI runs all three.
