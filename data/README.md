# MACS/CD4 data provenance

`raw/catdata_aids.csv` is a lossless CSV export of the `aids` object in CRAN
`catdata` 1.2.5.  The source archive is retained as
`raw/catdata_1.2.5.tar.gz`.

- Canonical source: <https://CRAN.R-project.org/package=catdata>
- Source archive SHA-256:
  `033fbfbe02ae790f682ebfcc1d6cd7afb9ecee9d1a434e0aadf45f1a40bd47bb`
- CSV SHA-256:
  `aa22da94c608e1db93e19c75acc52e6bb5e019ef27de81003f3ad2cf2916fd4a`
- Audit dimensions: 369 subjects and 2,376 observations.

The analysis retains raw CD4 count as the response because Hu, Huang and You
(2021, Section 6.2) state no response transformation.  Time is years from
seroconversion; `age` is age at seroconversion centered at 30 in the distributed
data; `cesd` is the time-varying depression score. Time, age, and CES-D are
linearly mapped to `[0,1]` using the fixed full-study support endpoints recorded
before the subject folds are drawn. The same coordinate map is reused in every
outer fold; no response-dependent scaling is performed.
