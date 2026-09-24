# Architecture

## Flow

```text
geo-vlms key=value ...
  │
  ├─ Hydra composes conf/config.yaml + overrides, checked against config.py
  │
  ├─ dataset builder (cfg.dataset._target_)  →  list[Example]
  │     uses the Task to phrase each prompt
  │
  ├─ backend (cfg.backend._target_)
  │
  ├─ provenance  →  <out>.meta.json
  │
  └─ run_inference: backend.generate() per Example  →  <out> (JSONL)

scripts/analyze.py <out>
  └─ Task.parse_response + Task.score per record  →  summary table
```

Inference and scoring are separate.
Records hold raw model output, so a parser or metric can change without rerunning the model.

## Modules

| Path | Role |
|---|---|
| `cli.py` | Entry point. Builds dataset and backend, writes provenance, runs inference. Handles `overwrite` and `resume`. |
| `config.py` | Structured config schemas and Hydra registration. |
| `conf/` | Default config and the `backend/` and `dataset/` groups. |
| `example.py` | `Example`: one (image, prompt, expected) question. |
| `tasks/` | `Task` base, `Counting`, `Existence`. Prompt text, parsing, metrics. |
| `datasets/` | `vhr10.py`, `dior.py`. Each exposes `build_dataset(data_dir, task, seed, ...)`. |
| `backends/` | `Backend` protocol, `Generation`, and the `huggingface` and `llama_server` implementations. |
| `inference.py` | Loops examples through a backend and writes records. |
| `runs.py` | Record writer and resume helpers. |
| `provenance.py` | Builds the `.meta.json` sidecar. |
| `analysis.py` | Loads, scores, and summarizes records with pandas. |
| `scripts/` | `analyze.py`, `plot_counting_density.py`, `prepare_dior.py`. |

## Units

One `Example` is one (image, category) question.
Every category is asked of every image, so a 10-category dataset gives 10 examples per image.
Most of them have expected count 0 or `False`.

## Selection by `_target_`

`cfg.backend` and `cfg.dataset` each name a Python callable in `_target_`.
`cli.py` calls it with `hydra.utils.instantiate`.
The dataset builder also gets `task` and `seed` from the CLI.
Nothing in `cli.py` names a concrete backend or dataset.
See [extending.md](extending.md).

Tasks are the exception.
They're looked up by name in `tasks.TASKS`.
