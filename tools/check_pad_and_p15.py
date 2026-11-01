import sys, json, numpy as np, torch
from causal_probe.eval import _load_model, PadToTrainingWidth
from causal_probe.scm import sample_random_scms
W = sys.argv[1]
m = _load_model("causal_probe/results/probe_intersample_L_seed0.ckpt", device="cpu")
inner = m.inner
def std(A): return (A - A.mean(0)) / A.std(0).clip(min=1e-9)
# (a) pad-noise sensitivity on the 5-feature reference
D = np.loadtxt(f"{W}/reference.csv", delimiter=",", skiprows=1); X, y = D[:, :5], D[:, 5]
Xs, ys = std(X), (y - y.mean()) / y.std()
outs = []
for seed in range(10):
    g = torch.Generator().manual_seed(seed)
    Xp = torch.cat([torch.tensor(Xs, dtype=torch.float32), torch.randn(512, 10, generator=g)], 1)[None]
    with torch.no_grad(): outs.append(inner(Xp, torch.tensor(ys, dtype=torch.float32)[None])[0, :5].numpy())
outs = np.array(outs)
print("pad-seed spread (5-feature reference): mean", np.round(outs.mean(0), 3), "sd", np.round(outs.std(0), 3), "min", np.round(outs.min(0), 3), "max", np.round(outs.max(0), 3))
# (b) 15-feature reference: no padding, so JS must match Python exactly
rng = np.random.default_rng(3)
scm = sample_random_scms(1, p=15, rng=rng, noise="laplace")[0]; s = scm.sample(512, rng)
names = [f"x{j+1}" for j in range(15)]
with open(f"{W}/reference15.csv", "w") as f:
    f.write(",".join(names + ["outcome"]) + "\n")
    for i in range(512): f.write(",".join(f"{v:.6g}" for v in list(s.X[i]) + [s.y[i]]) + "\n")
D = np.loadtxt(f"{W}/reference15.csv", delimiter=",", skiprows=1); X, y = D[:, :15], D[:, 15]
with torch.no_grad():
    t = inner(torch.tensor(std(X), dtype=torch.float32)[None], torch.tensor((y - y.mean()) / y.std(), dtype=torch.float32)[None])[0].numpy()
json.dump({"names": names, "tau_true": s.tau.tolist(), "tau_hat_python": t.tolist()}, open(f"{W}/reference15.json", "w"), indent=1)
print("p=15 python tau_hat:", np.round(t, 3)); print("p=15 true tau     :", np.round(s.tau, 3))
