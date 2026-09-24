# Records

Each run writes two files:
- `<out>`: JSONL, one record per example.
- `<out stem>.meta.json`: the provenance sidecar.

## Output path

Default:

```text
results/<dataset>/<task>/<model_name>/<backend>/records_seed<seed>.jsonl
```

`text_only=true` sends the same prompts with no image and adds a `_text_only` suffix.
`model_name` keeps its `/`, so `HuggingFaceTB/SmolVLM2-2.2B-Instruct` becomes two directories.
Override with `out=path.jsonl`.

A `--multirun` sweep over dataset, task, model, backend, seed, or `text_only` gets one path per job.
When sweeping any other key, set `out=` per job, or the second job stops on the existing file.

## Existing output

| Flags | Output exists | Behavior |
|---|---|---|
| none | no | Run. |
| none | yes | `FileExistsError`. |
| `overwrite=true` | either | Replace. |
| `resume=true` | yes | Skip ids already written, append the rest. |

`resume` requires the original `.meta.json` and refuses if any of these changed:
- `args.model_name`
- `args.max_new_tokens`
- `dataset.sha256`
- The backend's `describe()` output, except `base_url`.

A truncated last line from a hard kill is dropped and its example reruns.
Records are flushed one at a time, so a crash loses at most the one in flight.

## Record fields

```json
{
  "id": "pos/175:ship",
  "image_path": "data/vhr10/positive_image_set/175.jpg",
  "prompt": "How many ships are there in this image? Answer with a number only.",
  "expected": 0,
  "metadata": {"dataset": "vhr10", "split": "positive", "category": "ship"},
  "output": " 0",
  "model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
  "prompt_tokens": 1106,
  "completion_tokens": 3,
  "latency_s": null
}
```

| Field | Meaning |
|---|---|
| `id` | Stable example id. Resume keys on it. |
| `image_path` | Relative to the working directory. `null` for `text_only`. |
| `prompt` | Exact text sent. |
| `expected` | Ground truth. `int` for counting, `bool` for existence. |
| `metadata` | Dataset-specific. `analyze.py` flattens it into columns. |
| `output` | Raw model reply, unparsed. |
| `model_name` | Label from config. |
| `prompt_tokens`, `completion_tokens` | From the backend, if reported. |
| `latency_s` | Wall time of the request. `null` on huggingface. |
| `tokens` | Only with `top_logprobs`. Per token: `token`, `logprob`, and `top` (alternative → logprob). |

## Sidecar fields

| Field | Meaning |
|---|---|
| `command` | Command line. |
| `args` | Fully resolved config. |
| `started_at` | UTC ISO-8601. |
| `backend` | The backend's `describe()`: model revision, dtype, device, library or server build. |
| `dataset.num_examples` | Example count. |
| `dataset.sha256` | Hash of the sorted examples. Same hash means same questions and labels. |
| `env` | Python, geo_vlms version, platform. |
| `git` | Commit SHA and dirty flag, or `null` outside a checkout. |
| `resumes` | One entry per resume: `command`, `started_at`. |

## Analysis

```bash
uv run scripts/analyze.py \
  --task counting \
  --records <records> \
  [--groupby col ...] \
  [--metrics metric ...]
```

`--groupby` groups by any record or `metadata` column, e.g. `split`, `category`, `expected`.
`--metrics` picks metrics.
Metrics are defined in [tasks.md](tasks.md).

Counting only:

```bash
uv run scripts/plot_counting_density.py \
  --records <records> \
  --out figures/
```

This writes predicted-vs-expected scatter and per-count-bin error plots for positive rows.
