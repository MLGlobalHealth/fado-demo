"""Reference dataset for the web demo: a linear Laplace SCM (p=5, n=512) with known tau,
written as a raw-scale CSV, plus the Python pipeline's tau_hat on it (standardize -> pad to 15 with
N(0,1) -> model), and a bootstrap sigma, so the JS page can be checked against it."""
import json, sys, numpy as np, torch
from causal_probe.scm import sample_random_scms
from causal_probe.eval import _load_model
W = sys.argv[1]
for seed in range(1, 200):
    rng = np.random.default_rng(seed)
    scm = sample_random_scms(1, p=5, rng=rng, noise="laplace")[0]
    s = scm.sample(512, rng)
    if (np.abs(s.tau) > 0.15).sum() >= 2 and (np.abs(s.tau) < 1e-9).sum() >= 1:
        print("seed", seed); break
print(type(s).__name__, {k: getattr(s, k).shape if hasattr(getattr(s, k), "shape") else getattr(s, k) for k in ("X", "y", "tau") if hasattr(s, k)})
X, y, tau = np.asarray(s.X, float), np.asarray(s.y, float), np.asarray(s.tau, float)
# Rescale to arbitrary raw units so the page's standardisation is exercised.
scale = np.array([3.0, 0.5, 12.0, 1.0, 40.0]); shift = np.array([10, -2, 100, 0, 500])
Xr = X * scale + shift; yr = y * 7.5 + 20
names = ["age", "dose", "income", "score", "weight"]
with open(f"{W}/reference.csv", "w") as f:
    f.write(",".join(names + ["outcome"]) + "\n")
    for i in range(len(yr)):
        f.write(",".join(f"{v:.6g}" for v in list(Xr[i]) + [yr[i]]) + "\n")
m = _load_model("causal_probe/results/probe_intersample_L_seed0.ckpt", device="cpu")   # pads to 15 itself
def run(Xa, ya):
    Xs = (Xa - Xa.mean(0)) / Xa.std(0).clip(min=1e-9); ys = (ya - ya.mean()) / ya.std().clip(min=1e-9)
    with torch.no_grad():
        return m(torch.tensor(Xs, dtype=torch.float32)[None], torch.tensor(ys, dtype=torch.float32)[None])[0].numpy()
point = run(Xr, yr)
brng = np.random.default_rng(0); boots = []
for b in range(20):
    idx = brng.integers(0, len(yr), len(yr)); boots.append(run(Xr[idx], yr[idx]))
boots = np.array(boots)
out = {"names": names, "tau_true": tau.tolist(), "tau_hat_python": point.tolist(),
       "boot_mean": boots.mean(0).tolist(), "boot_sd": boots.std(0, ddof=1).tolist()}
json.dump(out, open(f"{W}/reference.json", "w"), indent=1)
for n_, t, p_, sd in zip(names, tau, point, boots.std(0, ddof=1)): print(f"{n_:8s} true={t:+.3f} hat={p_:+.3f} sd={sd:.3f}")
