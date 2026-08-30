# GEO-VLMs

Evaluates vision-language models on remote-sensing imagery.
It currently runs two tasks, existence ("Are there any ships in this image?") and counting ("How many ships are in this image?").
Models either load in-process from HuggingFace or sit behind a [llama-server](https://github.com/ggml-org/llama.cpp).

## Setup

```
uv sync
git config core.hooksPath .githooks
```

`uv sync` installs everything, including the `hf` extra (torch, transformers) that the huggingface backend needs.
A plain `pip install .` gives you the llama-server backend and the analysis tools.
Add `".[hf]"` to get the huggingface backend as well.

## Data

### NWPU VHR-10

Unpack [NWPU VHR-10](https://gcheng-nwpu.github.io/) under `data/vhr10` (gitignored):

```
data/vhr10/
  positive_image_set/    # 650 images with at least one annotated object
  negative_image_set/    # 150 images with none of the ten categories
  ground_truth/          # one bounding-box file per positive image
```

### DIOR

Download DIOR from the authors' [Google Drive folder](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) and unpack it under `data/dior` (gitignored):

```
data/dior/
  JPEGImages-trainval/
  JPEGImages-test/
  Annotations/
    Horizontal Bounding Boxes/
    Oriented Bounding Boxes/
  Main/
    train.txt
    val.txt
    test.txt
```

Prepare the dataset once by converting the horizontal XML annotations into the count table used by the DIOR loader:

```
uv run scripts/prepare_dior.py --data-dir data/dior
```

This writes `data/dior/counts.csv` with one row per image and category.
It verifies split membership, image and annotation coverage, filenames, and category names without modifying the downloaded files.

## Running an eval

```
uv run scripts/run.py dataset=vhr10 task=counting backend=huggingface model_name=Qwen/Qwen2.5-VL-3B-Instruct
```

Config is [Hydra](https://hydra.cc): `key=value` overrides on `conf/`, validated against the schemas in `src/geo_vlms/config.py`.
`--cfg job --resolve` prints the composed config without running.
Records default to `results/<dataset>/<task>/<model>/<backend>/records_seed<seed>.jsonl`; override with `out=`.
Each run writes one JSONL record per (image, category) pair, asking every category of every image so absence questions also come from images with objects.
VHR-10 knobs: `dataset.num_pos`, `dataset.num_neg`, `dataset.no_neg=true`.
DIOR knobs: `dataset.split` (default `test`), `dataset.num_images`.

Comma-separated values with `-m` sweep the cross product, one job at a time:

```
uv run scripts/run.py -m task=counting,existence dataset=vhr10,dior backend=huggingface model_name=Qwen/Qwen2.5-VL-3B-Instruct
```

Sweeps over dataset, task, model, backend, or seed each write to their own `results/` path; for any other axis, set `out=` per run or the exists check will stop the second job.
On a cluster, prefer one `run.py` invocation per configuration (e.g. one Slurm array task each) over `-m`, which runs jobs sequentially in a single process.

### llama-server

Start a server with a vision model (`--mmproj` is required) and point the run at it:

```
llama-server -m model.gguf --mmproj mmproj.gguf -c 16384 -ngl 99 --port 8080
uv run scripts/run.py dataset=vhr10 task=counting backend=llama_server backend.base_url=http://localhost:8080/v1 model_name=org/model-name
```

Here `model_name` only labels the records, and the run warns if it doesn't match what the server reports.
If the server requires auth, set `GEO_VLMS_LLAMA_API_KEY`; it defaults to `unused`.
Each request asks for greedy sampling with thinking off, so results are deterministic for a given GGUF and llama.cpp build.

## Container

```
docker build -t geo-vlms .
docker run --rm -v ./data:/app/data -v ./results:/app/results geo-vlms \
  python scripts/run.py dataset=vhr10 task=counting backend=llama_server backend.base_url=http://host:8080/v1 model_name=org/model-name
```

The image has the llama-server backend only, so it needs no GPU or torch.
The build args `BASE_IMAGE`, `PIP_INDEX_URL`, and `PIP_EXTRA_INDEX_URL` let you build from an internal base image and package mirror.

## Analyzing results

```
uv run scripts/analyze.py -t counting -r results/qwen25_counting.jsonl -g category
```

Scores each record and prints a summary table.
`-g` groups by any record columns such as `category` or `expected`, and `-m` picks which metrics to show.

## Development

```
uv run pytest
uv run ruff check .
uv run ruff format .
```

The pre-commit hook runs ruff, the pre-push hook also runs pytest, and CI runs all three.

### Reproducibility

Runs are meant to be deterministic on a given machine.
Decoding is greedy, sampling is seeded, and a `<out>.meta.json` sidecar records the run config.
To check, run a small eval twice and diff:

```
uv run scripts/run.py backend=huggingface model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct dataset.num_pos=3 dataset.no_neg=true out=/tmp/repro_a.jsonl
uv run scripts/run.py backend=huggingface model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct dataset.num_pos=3 dataset.no_neg=true out=/tmp/repro_b.jsonl
diff /tmp/repro_a.jsonl /tmp/repro_b.jsonl
```

The records files should be byte-identical.
The meta files should match apart from `command`, `args.out`, and `started_at`, and `dataset.sha256` and `backend.commit_hash` are the fields worth looking at.
Redo this after bumping torch or transformers.
The same check works against a llama-server.
