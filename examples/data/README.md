# Synthetic example data

`synthetic_survey.csv` is **synthetic**. It was produced by
[`../generate_synthetic_data.py`](../generate_synthetic_data.py) with a fixed random seed.
No participant exists, no real instrument is represented, and nothing in the file is an
empirical finding.

Contents: 200 invented participants (+1 deliberately duplicated row), two invented scales
(`sw1`-`sw6`, with `sw2` and `sw5` reverse-keyed; `sf1`-`sf4`) on a 1-5 response scale, plus
`age`, `group` and `screen_hours`.

Deliberate data-quality problems for demonstration: 12 missing item responses, two
out-of-range codes (7 and 9), and one duplicated `participant_id`.

Regenerate: `python examples/generate_synthetic_data.py`
