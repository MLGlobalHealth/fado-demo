"""Export a FADO checkpoint to ONNX (dynamic rows n AND dynamic feature count p) and check it against torch.

Usage, from the learning-to-explain repo root:
    PYTHONPATH=. uv run --with onnx --with onnxruntime --with onnxscript \
        python ../MLGHrepos/fado-demo/tools/export_onnx.py <ckpt> <out.onnx>

The current demo model is probe_foundation_intersample_L_p3_64_widthbatch_seed0_anygpu.ckpt
(trained with --width-mode batch on p in [3, 64], no padding), so the page feeds p columns
directly for 3 <= p <= 64. The earlier p<=15 model needed a fixed 15-column input.
"""
import sys, time, os, numpy as np, torch
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
from causal_probe.eval import _load_model
ckpt, out = sys.argv[1], sys.argv[2]
m = _load_model(ckpt, device="cpu")
a = (torch.load(ckpt, map_location="cpu", weights_only=False).get("args") or {})
print("ckpt args: width_mode=%s p_min=%s p_max=%s steps=%s" % (a.get("width_mode"), a.get("p_min"), a.get("p_max"), a.get("steps")))
inner = (m.inner if hasattr(m, "inner") else m).eval()
P_MIN, P_MAX = int(a.get("p_min") or 3), int(a.get("p_max") or 64)
print("inner:", type(inner).__name__, "params:", sum(p.numel() for p in inner.parameters()), "p range", P_MIN, P_MAX)
X = torch.randn(1, 256, 32); y = torch.randn(1, 256)
n = torch.export.Dim("n", min=8, max=4096)
p = torch.export.Dim("p", min=P_MIN, max=P_MAX)
with torch.no_grad():
    prog = torch.onnx.export(inner, (X, y), input_names=["X", "y"], output_names=["tau"],
                             dynamic_shapes={"X": {1: n, 2: p}, "y": {1: n}}, opset_version=18, dynamo=True)
prog.optimize()
prog.save(out)
import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(out))
sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
worst = 0.0
for pp, nn_ in ((3, 64), (5, 512), (15, 512), (15, 1000), (32, 512), (64, 64), (64, 512)):
    Xn = np.random.randn(1, nn_, pp).astype(np.float32); yn = np.random.randn(1, nn_).astype(np.float32)
    with torch.no_grad(): ref = inner(torch.from_numpy(Xn), torch.from_numpy(yn)).numpy()
    t = time.time(); got = sess.run(None, {"X": Xn, "y": yn})[0]; dt = time.time() - t
    d = float(np.abs(got - ref).max()); worst = max(worst, d)
    print(f"p={pp:2d} n={nn_:4d} max|diff|={d:.2e}  max|ref|={np.abs(ref).max():.2f}  ort_cpu={dt:.2f}s  out_shape={got.shape}")
print("onnx bytes:", os.path.getsize(out), " worst max|diff|:", f"{worst:.2e}")
