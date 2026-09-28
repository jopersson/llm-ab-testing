# Data provenance

## Source

The empirical application uses the Upworthy Research Archive, a public archive
of 32,487 randomized headline tests run by the media site Upworthy, released by
Matias, Munger, Aubin Le Quéré, and Ebersole in *Scientific Data* 8:195 (2021).
This work uses the archive's exploratory dataset, which is distributed through
OSF and is not included here. `Code/empirical/04_upworthy/prepare.py` prints the
download command and reads the file from `Code/empirical/output/data/`:

```bash
curl -sL -o upworthy_exploratory.csv https://osf.io/download/3vqmp/
```

Users of the archive should cite the paper above and observe the terms under
which the archive is distributed on OSF.

## What is redistributed here

Two derived tables and two completion caches are included so that the analysis
runs without re-downloading the archive or re-querying a language model.

| File | Contents |
|---|---|
| `Code/empirical/output/data/uw_experiments.parquet` | 7,206 headline variants across the 3,603 analyzed tests, with the randomized assignment, impressions, clicks, the click-through outcome, and the headline text, derived from the exploratory dataset by `prepare.py` |
| `Code/empirical/output/data/uw_with_ystar.parquet` | the same 7,206 rows with ten surrogate draws per variant appended as `Ystar_0` through `Ystar_9` |
| `Code/empirical/output/cache/uw.jsonl` | 72,060 raw `gpt-4o-mini` completions, one per variant and draw |
| `Code/empirical/output/cache/uw_contamination.jsonl` | completions for the headline-completion memorization probe |

The headline text is reproduced from the archive because the surrogate is a
function of it, so the analysis cannot be rerun without it. The surrogate draws
and the completion caches are outputs of this work rather than archive content.

## Model outputs

Surrogate outcomes were generated with `gpt-4o-mini` at temperature 0.7 with ten
draws per variant, under the prompt in
`Code/empirical/04_upworthy/prompts.py`. The caches are complete, so no API key
and no network access are needed to reproduce any result.
