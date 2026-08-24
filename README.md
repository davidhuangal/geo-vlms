# GEO-VLMs

A harness for evaluating vision-language models on remote-sensing imagery.
Right now it runs two tasks: existence ("Are there any ships in this image?") and counting ("How many ships are in this image?"), against any HuggingFace image-text-to-text model loaded in-process, or against a [llama-server](https://github.com/ggml-org/llama.cpp) instance over HTTP.

## Setup

```
uv sync
git config core.hooksPath .githooks   # enable pre-commit / pre-push hooks
```

The huggingface backend needs the `hf` extra (torch, transformers, and friends), which `uv sync` includes through the dev group.
A plain `pip install .` gives only the llama-server backend and the analysis tools; use `pip install ".[hf]"` for both.

## Data

Get the [NWPU VHR-10 dataset](https://gcheng-nwpu.github.io/) and unpack it under `data/vhr10` (the `data/` directory is gitignored):

```
data/vhr10/
  positive_image_set/    # 650 images, each containing at least one annotated object
  negative_image_set/    # 150 images containing none of the ten categories
  ground_truth/          # one bounding-box file per positive image
```

## Running an eval

```
uv run scripts/run_vhr10.py -t counting -m Qwen/Qwen2.5-VL-3B-Instruct -b huggingface -o results/qwen25_counting.jsonl
```

This builds one example per (image, category) pair, runs the model over all of them, and writes one JSONL record per example.
Positive images produce a question for every category, including ones absent from the image, so absence questions don't come only from the negative split.
Useful flags: `--num-pos` / `--num-neg` to subsample images, `--no-neg` to skip the negative set, `--device`, `--max-new-tokens`.

### Running against llama-server

Instead of loading a HuggingFace model in-process, the harness can act as a client to a llama.cpp server, which is how it runs on a GPU cluster.
Launch a llama-server with a vision model (the `--mmproj` projector is required), then point the run at it:

```
llama-server -m model.gguf --mmproj mmproj.gguf -c 16384 -ngl 99 --port 8080
uv run scripts/run_vhr10.py -t counting -m org/model-name -b llama-server --base-url http://localhost:8080/v1 -o results/model_counting.jsonl
```

`-m` does not load anything here; it labels the records, and the run warns when it doesn't match the model the server reports.
Greedy sampling and thinking-mode disabling are requested per-request, so results are deterministic for a fixed GGUF and llama.cpp build.

## Container

```
docker build -t geo-vlms .
docker run --rm -v ./data:/app/data -v ./results:/app/results geo-vlms \
  python scripts/run_vhr10.py -t counting -m org/model-name -b llama-server --base-url http://host:8080/v1 -o results/model_counting.jsonl
```

The image has only the llama-server backend; it needs no GPU or torch.
`BASE_IMAGE`, `PIP_INDEX_URL`, and `PIP_EXTRA_INDEX_URL` are build args for building from an internal base image and package mirror.

## Analyzing results

```
uv run scripts/analyze.py -t counting -r results/qwen25_counting.jsonl -g category
```

Parses and scores each record with the task's metrics and prints a summary table.
`-g` groups by any record columns (e.g. `category`, `expected`) and `-m` restricts which metrics are shown.

## Development

```
uv run pytest
uv run ruff check .
uv run ruff format .
```

The pre-commit hook runs ruff, the pre-push hook also runs pytest, and CI runs all three on every push and PR.

### Checking reproducibility

Inference is meant to be deterministic on a given machine: greedy decoding, seeded sampling, and a provenance sidecar (`<out>.meta.json`) recording the run config.
To verify end to end, run a tiny eval twice and diff:

```
uv run python scripts/run_vhr10.py -t counting -m HuggingFaceTB/SmolVLM2-2.2B-Instruct -b huggingface --num-pos 3 --no-neg -o /tmp/repro_a.jsonl
uv run python scripts/run_vhr10.py -t counting -m HuggingFaceTB/SmolVLM2-2.2B-Instruct -b huggingface --num-pos 3 --no-neg -o /tmp/repro_b.jsonl
diff /tmp/repro_a.jsonl /tmp/repro_b.jsonl
```

The records files should be byte-identical.
In the meta files, everything except `command`, `args.out`, and `started_at` should match.
In particular, `dataset.sha256` and `backend.commit_hash` should be identical.
Worth rerunning after bumping torch or transformers.
The same check applies to the llama-server backend against a running server.
