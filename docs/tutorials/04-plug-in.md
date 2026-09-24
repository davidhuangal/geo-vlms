# 4. Plug in a dataset and a backend

Add your own dataset and backend without touching the package.
Both live in one Python file and get picked up through a config dir.
Contracts: [extending.md](../extending.md).

Final layout, in any directory:

```text
my_plugins.py
my_conf/
  dataset/mine.yaml
  backend/canned.yaml
data/mine/
  labels.csv
  images/
```

## Part A: dataset

### Data

Any images plus a labels file.
Three VHR-10 images stand in here:

```bash
mkdir -p data/mine/images
cp data/vhr10/positive_image_set/{277,282,285}.jpg data/mine/images/
```

`data/mine/labels.csv` lists nonzero counts only:

```csv
image,category,count
277.jpg,ship,3
277.jpg,harbor,5
282.jpg,ship,1
282.jpg,harbor,8
285.jpg,ship,5
```

### Builder

`my_plugins.py`:

```python
import csv
from pathlib import Path

from geo_vlms.example import Example
from geo_vlms.tasks import Category, Counting, Existence, Task

CATEGORIES = [Category("ship", "ships"), Category("harbor", "harbors")]


def build_dataset(task: Task, seed: int, data_dir: str) -> list[Example]:
    if not isinstance(task, Counting | Existence):
        raise ValueError(f"Unsupported task: {type(task).__name__}")

    data_dir = Path(data_dir)
    with open(data_dir / "labels.csv") as f:
        counts = {
            (r["image"], r["category"]): int(r["count"]) for r in csv.DictReader(f)
        }

    examples = []
    for image in sorted({image for image, _ in counts}):
        for category in CATEGORIES:
            count = counts.get((image, category.name), 0)
            examples.append(
                Example(
                    id=f"{image}:{category.name}",
                    image_path=str(data_dir / "images" / image),
                    prompt=task.format_prompt(category),
                    expected=count if isinstance(task, Counting) else count > 0,
                    metadata={"dataset": "mine", "category": category.name},
                )
            )
    return examples
```

The CLI passes `task` and `seed`.
Every other argument comes from the yaml.
Every category is asked of every image, so `285.jpg:harbor` is an `expected=0` question.
This builder uses every image, so `seed` goes unused.

### Config

`my_conf/dataset/mine.yaml`:

```yaml
_target_: my_plugins.build_dataset
data_dir: data/mine
```

### Run

`PYTHONPATH=.` makes `my_plugins` importable:

```bash
PYTHONPATH=. uv run geo-vlms --config-dir my_conf \
  dataset=mine \
  backend=huggingface \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

```text
Built 6 mine counting examples. Using HuggingFaceTB/SmolVLM2-2.2B-Instruct via huggingface.
Wrote records to results/mine/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.jsonl
```

`analyze.py` works on these records unchanged:

```bash
uv run scripts/analyze.py \
  --task counting \
  --records results/mine/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.jsonl \
  --groupby category \
  --metrics exact_match absolute_error
```

```text
          exact_match  absolute_error
category
harbor       0.000000        4.333333
ship         0.333333        1.333333
```

## Part B: backend

A stub that answers from a lookup table.
It's handy for testing a pipeline without a model.

### Class

Add to `my_plugins.py`:

```python
from geo_vlms.backends.base import Generation


class CannedBackend:
    def __init__(self, model_name: str, answers: dict[str, str], default: str):
        self.model_name = model_name
        self.answers = dict(answers)
        self.default = default

    def generate(self, prompt, images, max_new_tokens, top_logprobs=None):
        if top_logprobs is not None:
            raise NotImplementedError("CannedBackend does not return logprobs.")

        for keyword, reply in self.answers.items():
            if keyword in prompt:
                return Generation(text=reply)
        return Generation(text=self.default)

    def describe(self):
        return {
            "kind": "canned",
            "name": self.model_name,
            "answers": self.answers,
            "default": self.default,
        }
```

`describe()` goes into `.meta.json`.
Put everything that changes outputs in it.
Here that's the whole lookup table.

`dict(answers)` converts Hydra's `DictConfig` to a plain dict so it serializes.

### Config

`my_conf/backend/canned.yaml`:

```yaml
_target_: my_plugins.CannedBackend
model_name: ${model_name}
answers:
  ships: "3"
  harbors: "There are 6 harbors."
default: "0"
```

`${model_name}` reuses the top-level value.

### Run

```bash
PYTHONPATH=. uv run geo-vlms --config-dir my_conf \
  dataset=mine \
  backend=canned \
  model_name=canned
```

```text
Built 6 mine counting examples. Using canned via canned.
Wrote records to results/mine/counting/canned/canned/records_seed0.jsonl
```

```bash
uv run scripts/analyze.py \
  --task counting \
  --records results/mine/counting/canned/canned/records_seed0.jsonl \
  --groupby category \
  --metrics exact_match absolute_error
```

```text
          exact_match  absolute_error
category
harbor       0.000000        3.000000
ship         0.333333        1.333333
```

`"There are 6 harbors."` parses as 6, the same as a bare `6`.

The built-in backends and datasets are still available alongside yours, such as `backend=canned dataset=vhr10`.
