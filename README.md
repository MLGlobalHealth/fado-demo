# FADO in the browser

A static, single-page demo of the FADO foundation model. The user drops in a CSV,
picks the outcome column, and gets per-feature causal effect estimates with
bootstrap uncertainty. Inference runs in the browser with onnxruntime-web
(WebGPU when available, WebAssembly otherwise); no data leaves the machine.

Everything here is static, so it can be served from GitHub Pages or any file host.

## Files

| File | What it is |
|---|---|
| `index.html` | The whole app: UI, CSV parsing, preprocessing, inference, chart, table. No build step. |
| `fado.onnx` | The wide FADO model (`probe_foundation_intersample_L_p3_64_widthbatch_seed0_anygpu.ckpt`, 14.2M parameters, trained on tables of 3 to 64 features with no padding), exported to ONNX with dynamic row and column dimensions. 58 MB, fp32. The paper's p≤15 model was the demo model until 10 Nov 2026; the wide model matches it at 15 features and extends to 64 (paper, appendix "Wider Tables"). |
| `reference.csv` | Sample data for the "Try the sample data" button: one simulated linear SCM, p=5, n=512, rescaled to arbitrary units. `reference.json` holds its true τ and the Python pipeline's τ̂. |
| `reference15.csv`, `reference64.csv` | 15- and 64-feature simulated datasets used to check that the browser pipeline matches Python exactly. The matching `.json` files hold the true τ and the Python τ̂. |
| `tools/export_onnx.py` | Exports a checkpoint to ONNX (dynamic rows and columns) and checks the ONNX output against PyTorch at several widths and row counts. Run from the `learning-to-explain` repo root with `PYTHONPATH=.`. |
| `tools/make_references.py` | Rewrites the Python-side reference estimates for the current model and builds `reference64.csv`. `tools/make_reference.py` and `tools/check_pad_and_p15.py` are the originals for the p≤15 model. |

External dependencies, loaded from CDNs and pinned: onnxruntime-web 1.20.1 (jsdelivr) and PapaParse 5.4.1 (cdnjs).

## What the page computes

For the selected feature columns `X` (at most 64, numeric) and outcome `y`:

1. Drop rows with any missing value among the selected columns. If more than 512 rows remain, take a random 512 (seeded).
2. Standardise every column to mean 0, population sd 1. This is the same convention as `causal_probe/real_datasets.py` and `causal_probe/fado_baseline.py`.
3. Feed the `p` columns directly. The model was trained on every width from 3 to 64, so no padding is needed; only `p` = 1 or 2 is padded with iid N(0,1) columns up to 3, and the first `p` outputs are kept.
4. Run the model. The output τ̂_i is the estimated total effect on `y` (in sd units of `y`) of shifting feature `i` by two sd, the paper's label definition `τ_i = A_yi · 2·sd(X_i)/sd(Y)`.
5. Bootstrap: resample rows with replacement, repeat 2–4, report the sd over replicates.

The table also reports `τ̂ · sd(y) / (2·sd(x_i))`, the implied slope in raw outcome units per raw unit of the feature.

## Verification done

Wide model (10 Nov 2026):

- ONNX vs PyTorch: max abs difference ≤ 1e-6 at p = 3, 5, 15, 32, 64 and n = 64 to 1000.
- Browser vs Python: `reference64.csv` (p = 64) max abs difference 2e-5 on WebGPU; `reference15.csv` (p = 15) and `reference.csv` (p = 5, now unpadded) max abs difference 7e-7. A 100-replicate bootstrap on `reference.csv` gives the same sd and 5/50/95% quantiles as Python to within resampling noise.
- Timing per forward pass at n = 512 on WebGPU (Apple M-series): about 2 s at p = 64 (0.6 s at p = 15, 0.1 s per bootstrap replicate at p = 5). Single-threaded WebAssembly takes about 26 s per pass at p = 64, during which the tab is unresponsive, so on the CPU path keep the feature count or the bootstrap count small. GitHub Pages does not send the cross-origin-isolation headers needed for multi-threaded WebAssembly, so the CPU path stays single-threaded there.

Earlier p≤15 model: ONNX vs PyTorch ≤ 1.4e-6; browser vs Python identical to four decimals at p = 15 on WebGPU and WebAssembly; at p = 5 the differences were within the spread of the padding noise. About 1 s per pass on WebGPU, 6.5 s single-threaded WebAssembly.

## Query parameters

- `?csv=<same-origin path>` preloads a dataset, e.g. `index.html?csv=reference15.csv`.
- `?ep=wasm` forces the CPU path (for testing).

## Local testing

```bash
python3 -m http.server 8791 --bind 127.0.0.1   # from this directory
open http://127.0.0.1:8791/index.html
```

`file://` does not work because the model is fetched.

## Deploying to GitHub Pages

Commit this directory to a public repository (the 58 MB model is under GitHub's 100 MB per-file limit; do not put it in Git LFS, Pages does not serve LFS objects) and enable Pages for the branch. No build step.

Live page: https://mlgh.net/fado-demo/
