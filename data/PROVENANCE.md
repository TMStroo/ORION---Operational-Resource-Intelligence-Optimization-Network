# Data provenance

No dataset is committed to this repository. Both public benchmark files are
downloaded on demand by `orion.data.benchmark_adapters` and cached under
`data/cache/`, which is git-ignored.

| Dataset | Source | Use | Licence / attribution |
|---|---|---|---|
| `jobshop1.txt` | OR-Library, Brunel University, `jobshop1.txt` (Fisher & Thompson; Taillard; Lawrence; Adams et al.) | job-shop benchmark, 82 instances | Free for research and teaching use; see the OR-Library page for terms. |
| Solomon instances (c101, r101) | SINTEF / VRP-REP, Solomon 100-customer VRPTW set | vehicle routing with time windows | Solomon (1987); widely redistributed. |

The download is keyless and requires no account. `load_jobshop` raises a
`MissingBenchmark` error naming the expected file if the cache is empty, so a
clean clone reports what it needs rather than failing obscurely.

Fetch them with:

```bash
PYTHONPATH=backend/src python -m orion benchmark
```

or by calling the loader directly, which downloads and caches on first use.
