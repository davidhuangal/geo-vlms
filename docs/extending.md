# Extending

Backends and datasets plug in from outside the package through a config dir.
Tasks are added inside the package.
For a worked example, see [tutorials/04-plug-in.md](tutorials/04-plug-in.md).

## Config dir

```text
my_conf/
  backend/<name>.yaml
  dataset/<name>.yaml
```

```bash
uv run geo-vlms --config-dir my_conf \
  backend=<name> \
  dataset=<name> \
  ...
```

Your yaml is layered on the package's `conf/`, so built-in choices still work.
Each yaml names a callable with `_target_`.
Its other keys become keyword arguments.
The module must be importable, for example via `PYTHONPATH` or an installed package.

## Backend

Any class with these two methods.
It doesn't need to subclass anything.

```python
from geo_vlms.backends.base import Generation


class MyBackend:
    def generate(
        self,
        prompt: str,
        images: list[str | bytes] | None,  # paths or encoded bytes; None for text_only
        max_new_tokens: int,
        top_logprobs: int | None = None,
    ) -> Generation: ...

    def describe(self) -> dict: ...
```

`Generation` fields: `text` (required), `tokens`, `prompt_tokens`, `completion_tokens`, `latency_s`.

`describe()` goes into `.meta.json` under `backend`.
Resume refuses if it changes, except `base_url`.
Put everything that affects outputs in it, like model revision, quantization, and sampling settings.
It must be JSON-serializable.

Should decode greedily.
Should raise on `top_logprobs` if unsupported.

```yaml
# my_conf/backend/mine.yaml
_target_: my_pkg.MyBackend
model_name: ${model_name}
```

## Dataset

A function called with `task` and `seed` plus the yaml's keys.

```python
from geo_vlms.example import Example
from geo_vlms.tasks import Category, Counting, Existence, Task

def build_dataset(task: Task, seed: int, data_dir: str, ...) -> list[Example]: ...
```

Rules:
- `id` is unique and stable across runs. Resume keys on it.
- `prompt` comes from `task.format_prompt(Category(name, plural))`.
- `expected` is an `int` for `Counting` and a `bool` for `Existence`.
- Raise on tasks you don't support.
- Sampling uses `seed`, and file lists are sorted, so the order is stable.
- `metadata` is JSON-serializable. Include `dataset` and `category`, which grouping and plots use.
- Ask every category of every image, not only the present ones.

```yaml
# my_conf/dataset/mine.yaml
_target_: my_pkg.build_dataset
data_dir: data/mine
```

Without a structured schema, key typos surface as a `TypeError` from your function, not at composition.
Built-in datasets register a schema in `config.py`.

## Task

In-package only.

1. Subclass `Task[PredT]` in `tasks/` and implement `format_prompt`, `parse_response`, and `score`.
2. `score` returns the same keys on every branch. Use NaN, not a missing key, for undefined metrics.
3. Register it in `tasks/__init__.py` `TASKS`.
4. Teach each dataset builder what `expected` is for it.
