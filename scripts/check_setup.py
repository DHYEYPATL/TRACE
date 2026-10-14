from __future__ import annotations
import argparse
import importlib.metadata as md
import sys
from pathlib import Path

# Ensure project root is on sys.path so assetwatch package is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check AssetWatch AI — Python, CUDA, OpenCV, identity cache and camera devices")
    parser.add_argument("--probe-cameras", action="store_true",
                        help="Try opening camera indices 0-5")
    args = parser.parse_args()

    ok = True
    print(f"Python: {sys.version.split()[0]} ({'64-bit' if sys.maxsize > 2**32 else '32-bit'})")

    # ── Package versions ───────────────────────────────────────────
    for package in ("torch", "torchvision", "ultralytics",
                    "opencv-contrib-python", "Flask", "numpy"):
        try:
            print(f"  {package}: {md.version(package)}")
        except md.PackageNotFoundError:
            print(f"  {package}: NOT INSTALLED")
            ok = False

    # ── PyTorch / CUDA ─────────────────────────────────────────────
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  CUDA available: {cuda}")
        if cuda:
            print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version (torch): {torch.version.cuda}")
        else:
            print("  Compute mode: CPU fallback")
    except Exception as exc:
        print(f"  PyTorch check failed: {exc}")
        ok = False

    # ── OpenCV + CSRT ──────────────────────────────────────────────
    try:
        import cv2
        tracker = (hasattr(cv2, "TrackerCSRT_create") or
                   (hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create")))
        print(f"  OpenCV: {cv2.__version__}  CSRT tracker: {tracker}")
        if not tracker:
            print("  WARNING: CSRT tracker missing — manual ROI tracking will not work.")
            ok = False
    except Exception as exc:
        print(f"  OpenCV check failed: {exc}")
        ok = False

    # ── Identity + storage layer ───────────────────────────────────
    try:
        from assetwatch.identity import GALLERY_MAX, ENRICH_MIN_CONF, ENRICH_DIVERSITY
        from assetwatch.storage import GALLERY_MAX as STORE_MAX
        print(f"  Identity: gallery_max={GALLERY_MAX}  enrich_conf={ENRICH_MIN_CONF}  diversity={ENRICH_DIVERSITY}")
        assert GALLERY_MAX == 10, f"Expected gallery_max=10, got {GALLERY_MAX}"
        assert STORE_MAX == GALLERY_MAX, "Storage and identity GALLERY_MAX mismatch"
        print("  Identity layer: OK")
    except Exception as exc:
        print(f"  Identity layer check failed: {exc}")
        ok = False

    # ── Monitoring mode ────────────────────────────────────────────
    try:
        import assetwatch.service as svc
        mode_info = {"always": False, "night": True}
        for m, expected_idle in mode_info.items():
            svc._MODE = m
            assert svc._mode_is_idle() == expected_idle, f"Mode {m}: expected idle={expected_idle}"
        svc._MODE = "always"
        print("  Monitoring mode logic: OK")
    except Exception as exc:
        print(f"  Monitoring mode check failed: {exc}")
        ok = False

    # ── Unit tests (quick) ─────────────────────────────────────────
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            capture_output=True, text=True, timeout=60)
        last = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
        print(f"  Unit tests: {last}")
        if result.returncode != 0:
            print(result.stdout[-800:])
            ok = False
    except Exception as exc:
        print(f"  Could not run unit tests: {exc}")

    # ── Camera probe ───────────────────────────────────────────────
    if args.probe_cameras:
        try:
            import cv2
            for index in range(6):
                found = False
                for backend in ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                                 if sys.platform == "win32" else [cv2.CAP_ANY]):
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            print(f"  Camera {index}: AVAILABLE ({w}x{h})")
                            cap.release()
                            found = True
                            break
                    cap.release()
                if not found:
                    print(f"  Camera {index}: not detected")
        except Exception as exc:
            print(f"  Camera probe failed: {exc}")

    print()
    if ok:
        print("All checks passed. Run: python app.py")
    else:
        print("Some checks failed — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
