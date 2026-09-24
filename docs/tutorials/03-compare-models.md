# 3. Compare models

Sweep two models over both tasks with and without images, then read the results side by side.
Models: `HuggingFaceTB/SmolVLM-Instruct` and `HuggingFaceTB/SmolVLM2-2.2B-Instruct`.
It takes about 25 minutes on an M5 Pro, plus a 4.2 GB download for SmolVLM.

## Sweep

```bash
uv run geo-vlms --multirun \
  backend=huggingface \
  model_name=HuggingFaceTB/SmolVLM-Instruct,HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  task=counting,existence \
  text_only=false,true \
  dataset.num_pos=10 \
  dataset.num_neg=3
```

`--multirun` runs the cross product, 2 × 2 × 2 = 8 jobs, one after another.
Each job asks the same 130 questions (13 images × 10 categories).
Every job writes to its own path:

```text
results/vhr10/{counting,existence}/HuggingFaceTB/{SmolVLM-Instruct,SmolVLM2-2.2B-Instruct}/huggingface/
  records_seed0.jsonl
  records_seed0_text_only.jsonl
```

`text_only=true` sends the same prompts with no image.
It measures what the model answers from the question alone.

On a cluster, run one `geo-vlms` per configuration, such as one Slurm array task each, instead of `--multirun`.

## Counting

```bash
uv run scripts/analyze.py \
  --task counting \
  --records <records> \
  --metrics valid exact_match absolute_error within_1
```

| Model | Input | exact_match | absolute_error | within_1 |
|---|---|---|---|---|
| SmolVLM | image | 0.69 | 2.68 | 0.88 |
| SmolVLM2 | image | 0.68 | 1.02 | 0.85 |
| SmolVLM | text only | 0.05 | 2.15 | 0.78 |
| SmolVLM2 | text only | 0.00 | 9.76 | 0.00 |

Exact match is about equal.
SmolVLM's error is higher because of one record:

```bash
uv run scripts/analyze.py \
  --task counting \
  --records <SmolVLM records> \
  --groupby split \
  --metrics exact_match absolute_error
```

```text
          exact_match  absolute_error
split
negative         0.80            0.20
positive         0.66            3.43
```

It answered `100.` for an image with 29 storage tanks (`pos/326:storage tank`).
With 130 rows, one outlier moves the mean error.
Check `exact_match` and `within_1` too, and look at the raw `output` for the worst rows.

Text only, SmolVLM2 replies `There are 10 airplanes in the image.` to nearly everything.
So it isn't reading counts from the question.

## Existence

Most questions have answer "no".
Only 13 of the 130 have an object present.
Always group by `expected`:

```bash
uv run scripts/analyze.py \
  --task existence \
  --records <records> \
  --groupby expected \
  --metrics correct
```

| Model | Input | correct, absent | correct, present | correct, all |
|---|---|---|---|---|
| SmolVLM | image | 0.82 | 0.92 | 0.83 |
| SmolVLM2 | image | 0.85 | 1.00 | 0.87 |
| either | text only | 1.00 | 0.00 | 0.90 |

Without an image, both models always answer "No".
That scores 0.90 overall, which beats both models with the image.
The overall number rewards saying "no".
Compare on the present column, and on absent rows from the negative split, because VHR-10 labels miss objects ([datasets.md](../datasets.md#label-noise)).

## Plot

Counting only:

```bash
uv run scripts/plot_counting_density.py \
  --records results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/huggingface/records_seed0.jsonl \
  --out figures/SmolVLM2
```

This writes `records_seed0-scatter.png`, predicted vs expected counts, and `records_seed0-bins.png`, error by expected count.
Both cover positive rows only.
Empty bins are left out of the bar plot.
It also prints the binned table:

```text
              n  mean_absolute_error  mean_relative_error  exact_match
expected_bin
1             6             0.000000             0.000000     1.000000
2             3             0.666667             0.333333     0.333333
3-5           2             2.000000             0.500000     0.000000
6-10          1             2.000000             0.285714     0.000000
11-20         0                  NaN                  NaN          NaN
21+           1            17.000000             0.586207     0.000000
```

13 positive rows is too few to conclude anything.
Drop `dataset.num_pos` for a full run.

Next: [4. Plug in a dataset and a backend](04-plug-in.md).
