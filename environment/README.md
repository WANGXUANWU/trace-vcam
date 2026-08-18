# Locked computation environment

The formal benchmark uses Python 3.12.13 with the exact packages in the root
`requirements.txt`.  Original R implementations use R 4.6.1 (UCRT) and the
packages recorded in `r-packages.lock.txt`.  The R library used for the audit
is isolated at `C:/Users/24481084/.cache/vcam-r/library-4.6.1`; no method is
silently replaced when an R dependency is unavailable.

The external Zhao--Sun--Yang implementation is pinned separately by its Git
commit and source hashes under `benchmarks/vendor/`.

