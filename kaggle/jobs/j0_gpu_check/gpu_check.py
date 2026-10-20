"""J0: report whether Kaggle gave us a GPU. Writes <out>/gpu.json."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/kaggle/working/out")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--data-root", default=None)
ap.add_argument("--device", default="auto")
ap.add_argument("--compute-label", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
import torch  # noqa: E402

info = {"cuda_available": torch.cuda.is_available(), "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
json.dump(info, open(os.path.join(a.out, "gpu.json"), "w"), indent=1)
print(info)
