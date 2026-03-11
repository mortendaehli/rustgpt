# Open Dataset Notes

These files are the local open-dataset path for the recommended RustGPT workflow.

## Pretraining

Source dataset:

- `roneneldan/TinyStories`
- dataset page: `https://huggingface.co/datasets/roneneldan/TinyStories`

Downloaded files:

- `tinystories/train-00000-of-00004.parquet`
  - one official training shard
  - about `237 MB`
- `tinystories/validation.parquet`
  - official validation split
  - about `9.5 MB`

Validation:

- RustGPT successfully normalized `validation.parquet` through `prepare-data` as `parquet-text`
- local smoke commands in the main `README.md` were run against this dataset

## Chat SFT

Source dataset:

- `HuggingFaceTB/smoltalk`
- dataset page: `https://huggingface.co/datasets/HuggingFaceTB/smoltalk`
- local subset used here: `data/everyday-conversations`

Why this subset:

- it is small enough for local classroom SFT
- it already normalizes cleanly to RustGPT's `messages` schema
- it is a practical small-chat dataset for the first end-to-end workflow

Downloaded files:

- `smoltalk-everyday/train.parquet`
  - `2260` records
  - about `924 KB`
- `smoltalk-everyday/test.parquet`
  - `119` records
  - about `51 KB`

Validation:

- RustGPT successfully normalized `train.parquet` through `prepare-data` as `parquet-chat`
- local smoke commands in the main `README.md` were run against this dataset

## Why Not Download Everything?

The upstream datasets are larger than what is useful for a repo that students clone locally.

This repo intentionally keeps:

- one real TinyStories shard for pretraining
- one small real chat dataset for SFT

That is enough to teach the complete local workflow without turning the repository into a multi-gigabyte data dump.
