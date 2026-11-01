"""Export the FADO foundation model (inner module, p=15 inputs, dynamic n) to ONNX and check it against torch."""
import sys, time, os, numpy as np, torch
torch.backends.mha.set_fastpath_enabled(False)
from causal_probe.eval import _load_model
out = sys.argv[1]
m = _load_model("causal_probe/results/probe_intersample_L_seed0.ckpt", device="cpu")
inner = m.inner.eval()
print("inner:", type(inner).__name__, "params:", sum(p.numel() for p in inner.parameters()))
X = torch.randn(1, 256, 15); y = torch.randn(1, 256)
n = torch.export.Dim("n", min=8, max=4096)
with torch.no_grad():
    prog = torch.onnx.export(inner, (X, y), input_names=["X", "y"], output_names=["tau"],
                             dynamic_shapes={"X": {1: n}, "y": {1: n}}, opset_version=18, dynamo=True)
prog.optimize()
prog.save(out)
import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(out))
sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
for nn_ in (64, 256, 512, 1000):
    Xn = np.random.randn(1, nn_, 15).astype(np.float32); yn = np.random.randn(1, nn_).astype(np.float32)
    with torch.no_grad(): ref = inner(torch.from_numpy(Xn), torch.from_numpy(yn)).numpy()
    t = time.time(); got = sess.run(None, {"X": Xn, "y": yn})[0]; dt = time.time() - t
    print(f"n={nn_:4d} max|diff|={np.abs(got-ref).max():.2e}  max|ref|={np.abs(ref).max():.2f}  ort_cpu={dt:.2f}s")
print("onnx bytes:", os.path.getsize(out))
