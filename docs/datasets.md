# Datasets

Select with `dataset=vhr10` (default), `dataset=dior`, or `dataset=dota`.
Data lives under `data/` (gitignored).
All datasets support both tasks.

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
uv run geo-vlms prepare-dior --data-dir data/dior
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

## DOTA-v1.5

Source: [captain-whu.github.io/DOTA](https://captain-whu.github.io/DOTA/dataset.html).
Get the images and the oriented labels (`DOTA-v1.5_train.zip`, `DOTA-v1.5_val.zip`), not the `_hbb` ones.

```text
data/dota/
  train/images/P*.png
  train/labelTxt/P*.txt
  val/images/P*.png
  val/labelTxt/P*.txt
```

Prepare once:

```bash
uv run geo-vlms prepare-dota --data-dir data/dota
```

This cuts each image into 896 px tiles with no overlap and writes them to `data/dota/tiles/<split>/`.
The last tile in each row and column shifts back to end at the image border.
Images smaller than a tile are padded with black.
Pass `--splits train val` to tile both, `--tile-size` to change the size, and `--workers` to set the process count.
Re-running replaces the split's tiles.

It also writes two tables next to the tiles.
`tiles.csv` has one row per tile: `tile_id`, `image_id`, `tile_path`, `x0`, `y0`, `image_source` (GoogleEarth, GF, or JL), `gsd` (empty when unknown), and `pad_frac`, the share of the tile that is padding.
`objects.csv` has one row per object per tile it overlaps.
Its columns are `tile_id`, `obj_idx` (line order in the label file), `raw_category`, `category`, `coverage`, `difficult`, and the full object's `area_px`, `long_px`, `short_px`, `angle` of the long side in degrees, and center `cx`, `cy` in tile pixels.
`coverage` is the fraction of the object's polygon inside the tile.

Tiling cuts objects, so ground truth depends on how much of an object must be in view.
An object counts when `coverage >= dataset.min_cover`, 0.5 by default.
At 0.5 a cut object counts in the tile holding most of it.
Difficult objects are optional, so counting expects a `[lo, hi]` range when a tile has any.
Existence is `True` when any object counts, difficult or not.

16 categories, with names normalized for prompts (`soccer-ball-field` → `soccer ball field`).
`metadata.raw_category` keeps the original name.
`metadata.image_id`, `x0`, and `y0` locate the tile in its source image.
`metadata` also has the tile's `image_source`, `gsd`, and `pad_frac`.
`metadata.coverage` and `difficult` list every object of the category that overlaps the tile, so `geo-vlms analyze --min-cover` can rescore at another threshold without rerunning inference.

| Key | Default | Meaning |
|---|---|---|
| `dataset.data_dir` | `data/dota` | Dataset root. |
| `dataset.split` | `val` | `train` (1411) or `val` (458 images). Test labels are not public. |
| `dataset.num_tiles` | all | Tiles to sample from the split. |
| `dataset.categories` | all | Restrict to these normalized names. |
| `dataset.min_cover` | `0.5` | Fraction of an object that must be in the tile to count. |

Example ids look like `val/P0003_896_0:ship`.

## Keys are per dataset

Each dataset has its own schema.
Passing a key from the other one, like `dataset=dior dataset.num_pos=5`, fails at composition.
