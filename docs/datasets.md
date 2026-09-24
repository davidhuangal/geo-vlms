# Datasets

Select with `dataset=vhr10` (default) or `dataset=dior`.
Data lives under `data/` (gitignored).
Both datasets support both tasks.

## NWPU VHR-10

Source: [gcheng-nwpu.github.io](https://gcheng-nwpu.github.io/).

```text
data/vhr10/
  positive_image_set/    # 650 images, at least one annotated object
  negative_image_set/    # 150 images, none of the ten categories
  ground_truth/          # <image>.txt per positive image: (x1,y1),(x2,y2),category_id
```

Categories: airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle.

| Key | Default | Meaning |
|---|---|---|
| `dataset.data_dir` | `data/vhr10` | Dataset root. |
| `dataset.num_pos` | all | Positive images to sample. |
| `dataset.num_neg` | all | Negative images to sample. |
| `dataset.no_neg` | `false` | Skip negatives. |

Example ids look like `pos/175:ship` and `neg/012:harbor`.
`metadata.split` is `positive` or `negative`.
Sampling is seeded by `seed` and is independent for positives and negatives.

### Label noise

Annotations are incomplete, worst for vehicles.
Many images contain unannotated cars, so their "no vehicle" label is wrong.
A stronger model can score lower on absent categories because it sees them.
Don't take absent-category accuracy at face value.
Compare models on `expected=True` rows and on the negative split.

## DIOR

Source: the authors' [Google Drive folder](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC).

```text
data/dior/
  JPEGImages-trainval/
  JPEGImages-test/
  Annotations/Horizontal Bounding Boxes/
  Main/{train,val,test}.txt
```

Prepare once:

```bash
uv run scripts/prepare_dior.py --data-dir data/dior
```

This validates the download and writes `data/dior/counts.csv`, one row per (image, category).
The loader reads only this file.

20 categories, with names normalized for prompts (`Expressway-toll-station` → `expressway toll station`).
`metadata.raw_category` keeps the original name.

| Key | Default | Meaning |
|---|---|---|
| `dataset.data_dir` | `data/dior` | Dataset root. |
| `dataset.split` | `test` | `train` (5862), `val` (5863), or `test` (11738 images). |
| `dataset.num_images` | all | Images to sample from the split. |
| `dataset.categories` | all | Restrict to these normalized names. |

Example ids look like `test/11726:ship`.

## Keys are per dataset

Each dataset has its own schema.
Passing a key from the other one, like `dataset=dior dataset.num_pos=5`, fails at composition.
