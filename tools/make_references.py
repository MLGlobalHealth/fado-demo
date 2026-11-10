"""Python-side reference estimates for the browser checks, for the current demo model.

Usage, from the learning-to-explain repo root:
    PYTHONPATH=. uv run python ../MLGHrepos/fado-demo/tools/make_references.py <ckpt> <fado-demo dir>

Rewrites reference.json (p=5 sample data: point estimate + 20-replicate bootstrap) and
reference15.json (p=15 check) for the existing CSVs, and creates reference64.csv/.json
(a 64-feature linear Laplace SCM with the fixed-density edge probability 0.35*14/63 so
each node has ~2.5 parents, n=512). With the p<=64 model no input is padded, so every
browser estimate must match these to float precision.
"""
import json, sys, numpy as np, torch
torch.set_num_threads(4)
from causal_probe.eval import _load_model
from causal_probe.scm import sample_random_scms
ckpt, W = sys.argv[1], sys.argv[2]
m = _load_model(ckpt, device="cpu")
assert not hasattr(m, "inner"), "expected an unpadded (width-mode batch) checkpoint"
def std(A): return (A - A.mean(0)) / A.std(0).clip(min=1e-9)
def run(X, y):
    with torch.no_grad():
        return m(torch.tensor(std(X), dtype=torch.float32)[None],
                 torch.tensor((y - y.mean()) / y.std().clip(min=1e-9), dtype=torch.float32)[None])[0].numpy()

# p=5 sample data (CSV unchanged; tau_true kept from the existing json)
D = np.loadtxt(f"{W}/reference.csv", delimiter=",", skiprows=1); X, y = D[:, :5], D[:, 5]
old = json.load(open(f"{W}/reference.json"))
point = run(X, y)
brng = np.random.default_rng(0); boots = []
for b in range(20):
    idx = brng.integers(0, len(y), len(y)); boots.append(run(X[idx], y[idx]))
boots = np.array(boots)
json.dump({"names": old["names"], "tau_true": old["tau_true"], "tau_hat_python": point.tolist(),
           "boot_mean": boots.mean(0).tolist(), "boot_sd": boots.std(0, ddof=1).tolist()},
          open(f"{W}/reference.json", "w"), indent=1)
print("p=5  true", np.round(old["tau_true"], 3), "\n     hat ", np.round(point, 3), "\n     sd  ", np.round(boots.std(0, ddof=1), 3))

# p=15 check (CSV unchanged)
D = np.loadtxt(f"{W}/reference15.csv", delimiter=",", skiprows=1); X, y = D[:, :15], D[:, 15]
old = json.load(open(f"{W}/reference15.json"))
t = run(X, y)
json.dump({"names": old["names"], "tau_true": old["tau_true"], "tau_hat_python": t.tolist()},
          open(f"{W}/reference15.json", "w"), indent=1)
print("p=15 true", np.round(old["tau_true"], 2), "\n     hat ", np.round(t, 2))

# p=64 check: new dataset
rng = np.random.default_rng(64)
scm = sample_random_scms(1, p=64, rng=rng, noise="laplace", edge_prob=0.35 * 14 / 63)[0]; s = scm.sample(512, rng)
names = [f"x{j+1}" for j in range(64)]
with open(f"{W}/reference64.csv", "w") as f:
    f.write(",".join(names + ["outcome"]) + "\n")
    for i in range(512): f.write(",".join(f"{v:.6g}" for v in list(s.X[i]) + [s.y[i]]) + "\n")
D = np.loadtxt(f"{W}/reference64.csv", delimiter=",", skiprows=1); X, y = D[:, :64], D[:, 64]
t = run(X, y)
json.dump({"names": names, "tau_true": s.tau.tolist(), "tau_hat_python": t.tolist()}, open(f"{W}/reference64.json", "w"), indent=1)
nz = np.abs(s.tau) > 1e-9
print(f"p=64 nonzero tau: {nz.sum()}/64; pearson(hat,true)={np.corrcoef(t, s.tau)[0,1]:.3f}; max|hat| on zero features={np.abs(t[~nz]).max():.3f}")
print("     true", np.round(s.tau[nz], 2), "\n     hat ", np.round(t[nz], 2))
