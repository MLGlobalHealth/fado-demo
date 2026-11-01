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
| `fado.onnx` | The foundation model (`probe_intersample_L_seed0.ckpt`, 14.2M parameters), exported to ONNX with a dynamic row dimension and a fixed 15-column input. 58 MB, fp32. |
| `reference.csv` | Sample data for the "Try the sample data" button: one simulated linear SCM, p=5, n=512, rescaled to arbitrary units. `reference.json` holds its true τ and the Python pipeline's τ̂. |
| `reference15.csv` | A 15-feature simulated dataset used to check that the browser pipeline matches Python exactly (no padding is involved at p=15). `reference15.json` holds the Python τ̂. |
| `tools/export_onnx.py` | Exports the checkpoint to ONNX and checks the ONNX output against PyTorch at several row counts. Run from the `learning-to-explain` repo root with `PYTHONPATH=.`. |
| `tools/make_reference.py`, `tools/check_pad_and_p15.py` | Build the reference datasets and their Python-side estimates. |

External dependencies, loaded from CDNs and pinned: onnxruntime-web 1.20.1 (jsdelivr) and PapaParse 5.4.1 (cdnjs).

## What the page computes

For the selected feature columns `X` (at most 15, numeric) and outcome `y`:

1. Drop rows with any missing value among the selected columns. If more than 512 rows remain, take a random 512 (seeded).
2. Standardise every column to mean 0, population sd 1. This is the same convention as `causal_probe/real_datasets.py` and `causal_probe/fado_baseline.py`.
3. If fewer than 15 features, pad with iid N(0,1) columns up to 15 and keep only the first `p` outputs, as `PadToTrainingWidth` in `causal_probe/eval.py` does.
4. Run the model. The output τ̂_i is the estimated total effect on `y` (in sd units of `y`) of shifting feature `i` by two sd, the paper's label definition `τ_i = A_yi · 2·sd(X_i)/sd(Y)`.
5. Bootstrap: resample rows with replacement, repeat 2–4, report the sd over replicates.

The table also reports `τ̂ · sd(y) / (2·sd(x_i))`, the implied slope in raw outcome units per raw unit of the feature.

## Verification done

- ONNX vs PyTorch: max abs difference ≤ 1.4e-6 at n = 64, 256, 512, 1000.
- Browser vs Python on `reference15.csv` (p = 15, no padding): identical to four decimals on both WebGPU and WebAssembly.
- Browser vs Python on `reference.csv` (p = 5, padded): differences are within the spread caused by different padding noise draws (Python across 10 pad seeds: dose 0.15–0.22, income 2.09–2.33; browser across 10 pad seeds: dose 0.17–0.28, income 2.14–2.29). The browser's Gaussian generator checks out (mean 0.002, variance 1.002, kurtosis 2.99 over 200k draws).
- Timing per forward pass at n = 512: about 1 s on WebGPU (Apple M-series), about 6.5 s single-threaded WebAssembly. GitHub Pages does not send the cross-origin-isolation headers needed for multi-threaded WebAssembly, so the CPU path stays single-threaded there.

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

Live page: https://mlglobalhealth.github.io/fado-demo/
