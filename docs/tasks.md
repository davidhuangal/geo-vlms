# Tasks

Select with `task=counting` (default) or `task=existence`.
A task phrases the prompt, parses the reply, and scores it.
Parsing and scoring run in `geo-vlms analyze`, not during inference.

## Prompts

Each task has a default prompt, shown below.
Replace it with `prompt`, a template filled with `{plural}` and `{name}`:

```bash
uv run geo-vlms task=counting \
  backend=huggingface \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  out=results/smolvlm2-counting-answer0.jsonl \
  "prompt='How many {plural} are there in this image? Answer 0 if there are none.'"
```

Hydra needs the inner quotes, because `{` otherwise starts a dict.
The template must use `{plural}` or `{name}`, and no other fields.
Records store the filled-in prompt, and `.meta.json` stores the template.
Resume refuses a changed prompt, since the dataset hash covers every prompt.
Pass a different `out=` for each prompt variant.
Parsing doesn't change, so keep the answer format the task expects.

## Counting

Prompt:

```text
How many {plural} are there in this image? Answer with a number only.
```

Parsing:
- The first integer in the reply, commas stripped (`"1,200"` → 1200).
- Otherwise the first number word from zero to twenty, or `none` (0).
- Otherwise `None` (invalid).

Metrics per record:

| Metric | Meaning |
|---|---|
| `valid` | 1 if parsed. |
| `exact_match` | 1 if prediction equals expected, or falls in its range. |
| `absolute_error` | `abs(pred - expected)`. |
| `signed_error` | `pred - expected`. Positive means overcount. |
| `relative_error` | `absolute_error / expected`. NaN when expected is 0, so its mean is MAPE over non-empty rows. |
| `within_1` | 1 if off by at most one. |

`expected` can be a `[lo, hi]` range, as in DOTA tiles with difficult objects.
Errors then use the nearest bound as `expected`.

Invalid replies score 0 on `exact_match` and `within_1` and NaN on the error metrics.
Means of error metrics are therefore over valid rows only.
Read them next to `valid`.

## Existence

Prompt:

```text
Are there any {plural} in this image? Answer yes or no.
```

Parsing takes the first standalone `yes`, `no`, `y`, or `n`, case-insensitive.

| Metric | Meaning |
|---|---|
| `valid` | 1 if parsed. |
| `correct` | 1 if prediction equals expected. NaN when invalid. |

`correct` averages over valid rows only.
Group by `expected` to split recall on present objects from accuracy on absent ones:

```bash
uv run geo-vlms analyze \
  --records <records> \
  --groupby expected
```

## Categories

Datasets pass a `Category(name, plural)` to `format_prompt`.
`name` is singular and goes in example ids and `metadata.category`.
`plural` goes in the prompt.
