# 1. First eval

Run SmolVLM2 on a few VHR-10 images, look at the output, and score it.
It takes a few minutes on a laptop after the model download (about 4.5 GB).

## Setup

```bash
uv sync
```

Unpack VHR-10 under `data/vhr10` as described in [datasets.md](../datasets.md).

## Run

```bash
uv run geo-vlms \
  backend=huggingface \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  task=counting \
  dataset.num_pos=5 \
  dataset.num_neg=2
```

This samples 5 positive and 2 negative images.
It asks each image how many of each of the 10 categories it has, so there are 70 questions.

```text
Built 70 vhr10 counting examples. Using HuggingFaceTB/SmolVLM2-2.2B-Instruct via huggingface.
Wrote run provenance to results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.meta.json
Running Inference.: 100%|████████| 70/70 [01:44<00:00,  1.50s/it]
Wrote records to results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.jsonl
```

To print the composed config without running, add `--cfg job --resolve`.

## Look at the output

```bash
R=results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.jsonl
head -n 1 $R | python -m json.tool
```

```json
{
    "id": "pos/175:airplane",
    "image_path": "data/vhr10/positive_image_set/175.jpg",
    "prompt": "How many airplanes are there in this image? Answer with a number only.",
    "expected": 0,
    "metadata": {"dataset": "vhr10", "split": "positive", "category": "airplane"},
    "output": " 0",
    "model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "prompt_tokens": 1106,
    "completion_tokens": 3,
    "latency_s": null
}
```

`output` is the raw reply.
Nothing is parsed or scored yet.

The `.meta.json` next to it records the resolved config, the model's commit hash, dtype, device, library versions, and a hash of the 70 questions.
Field reference: [records.md](../records.md).

## Score

```bash
uv run scripts/analyze.py \
  --task counting \
  --records $R \
  --metrics valid exact_match absolute_error within_1
```

```text
      valid  exact_match  absolute_error  within_1
mean    1.0     0.671429        0.714286  0.842857
```

Group by any record or metadata column:

```bash
uv run scripts/analyze.py \
  --task counting \
  --records $R \
  --groupby category \
  --metrics exact_match signed_error
```

```text
                    exact_match  signed_error
category
airplane               1.000000      0.000000
baseball diamond       0.428571      0.857143
...
vehicle                0.428571      2.714286
```

A positive `signed_error` means the model overcounts.
Be careful with vehicles, because VHR-10 misses many of them in its labels ([datasets.md](../datasets.md#label-noise)).
Metric definitions: [tasks.md](../tasks.md).

## Rerun

Running the same command again fails, because the records file exists:

```text
FileExistsError: results/.../records_seed0.jsonl already exists; ...
```

Pass `overwrite=true` to replace it.
To finish an interrupted run, pass `resume=true`.
To keep both, pass a new `out=`.

Next: [2. llama-server](02-llama-server.md).
