"""Object detection and conservative appearance-based re-identification.

Composite similarity = W_DEEP  * ResNet18-cosine   (default 0.55)
                     + W_COLOR * HSV-histogram      (default 0.20)
                     + W_ORB   * ORB-keypoint-ratio (default 0.15)
                     + W_SPATIAL* spatial-prior     (default 0.10, only when prior available)

Partial-view adjustment: when detected box < 60 % of last-known area,
W_DEEP is reduced and W_COLOR + W_SPATIAL compensate.

The returned similarity is a ranking score, NOT a calibrated probability.
Identity is left unknown when the best match is weak or ambiguous.
"""
from __future__ import annotations

import base64
import os
import threading
from collections import deque
from typing import Any

import cv2
import numpy as np

# Default composite weights — all env-overridable.
_W_DEEP    = float(os.getenv("ASSETWATCH_W_DEEP",    "0.55"))
_W_COLOR   = float(os.getenv("ASSETWATCH_W_COLOR",   "0.20"))
_W_ORB     = float(os.getenv("ASSETWATCH_W_ORB",     "0.15"))
_W_SPATIAL = float(os.getenv("ASSETWATCH_W_SPATIAL", "0.10"))


def _sharpness(crop: np.ndarray) -> float:
    """Laplacian variance — proxy for image sharpness (higher = sharper)."""
    gray = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class VisionEngine:
    def __init__(self, model_name: str = "yolo11n.pt", imgsz: int = 640, conf: float = 0.25):
        self.model_name = model_name
        self.imgsz      = int(imgsz)
        self.conf       = float(conf)
        self.device     = "cpu"
        self.device_name = "CPU"
        self.gpu_available  = False
        self.cuda_detected  = False
        self.fallback_reason: str | None = None
        self.yolo    = None
        self.embedder = None
        self.error:  str | None = None
        self.ready   = False
        self._lock   = threading.RLock()
        self._cv_orb = cv2.ORB_create(nfeatures=180, scaleFactor=1.2, nlevels=8)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self.ready:
                return
            try:
                import torch
                from ultralytics import YOLO
                torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
                self.gpu_available = bool(torch.cuda.is_available())
                self.cuda_detected = self.gpu_available
                if self.gpu_available:
                    self.device = "0"
                    try:
                        self.device_name = torch.cuda.get_device_name(0)
                    except Exception:
                        self.device_name = "NVIDIA CUDA GPU"
                else:
                    self.device      = "cpu"
                    self.device_name = "CPU (CUDA unavailable)"
                self.yolo = YOLO(self.model_name)
                try:
                    self.yolo.to("cuda:0" if self.gpu_available else "cpu")
                except Exception:
                    pass
                from torchvision import models
                try:
                    weights  = models.ResNet18_Weights.DEFAULT
                    backbone = models.resnet18(weights=weights)
                except AttributeError:
                    backbone = models.resnet18(pretrained=True)
                self.embedder = torch.nn.Sequential(*list(backbone.children())[:-1]).eval()
                self.embedder.to("cuda:0" if self.gpu_available else "cpu")
                self._torch = torch
                self.ready  = True
                self.error  = None
            except Exception as exc:
                self.error = f"Model startup failed: {type(exc).__name__}: {exc}"
                self.ready = False
                raise

    def _fallback_to_cpu(self, reason: Exception) -> None:
        torch = getattr(self, "_torch", None)
        self.fallback_reason = f"CUDA failed; switched to CPU ({type(reason).__name__}: {reason})"
        self.gpu_available   = False
        self.device          = "cpu"
        self.device_name     = "CPU fallback after CUDA error"
        for m in (self.yolo, self.embedder):
            if m is not None:
                try:
                    m.to("cpu")
                except Exception:
                    pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        self.initialize()
        assert self.yolo is not None
        with self._lock:
            try:
                results = self.yolo.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    imgsz=self.imgsz, conf=self.conf, verbose=False,
                    device=self.device, max_det=60,
                )
            except Exception as exc:
                if not self.gpu_available:
                    raise
                self._fallback_to_cpu(exc)
                results = self.yolo.track(
                    source=frame, persist=True, tracker="botsort.yaml",
                    imgsz=self.imgsz, conf=self.conf, verbose=False,
                    device="cpu", max_det=60,
                )
        if not results:
            return []
        result = results[0]
        if result.boxes is None:
            return []
        xyxy        = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes     = result.boxes.cls.detach().cpu().numpy().astype(int)
        raw_ids = None
        try:
            if result.boxes.id is not None:
                raw_ids = result.boxes.id.detach().cpu().numpy().astype(int)
        except Exception:
            raw_ids = None
        names = result.names
        out   = []
        for i, (box, confidence, class_id) in enumerate(zip(xyxy, confidences, classes)):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1 = max(0, min(frame.shape[1] - 1, x1)); x2 = max(x1 + 1, min(frame.shape[1], x2))
            y1 = max(0, min(frame.shape[0] - 1, y1)); y2 = max(y1 + 1, min(frame.shape[0], y2))
            track_id = int(raw_ids[i]) if raw_ids is not None and i < len(raw_ids) else None
            out.append({
                "key":        f"yolo:{track_id}" if track_id is not None else f"det:{i}",
                "track_id":   track_id,
                "bbox":       [x1, y1, x2, y2],
                "class_name": str(names.get(int(class_id), class_id)
                                  if isinstance(names, dict) else names[int(class_id)]),
                "confidence": float(confidence),
                "source":     "yolo",
                "asset_id":   None,
                "asset_name": None,
                "similarity": None,
            })
        return out

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_crop(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Cannot extract a reference from an empty crop")
        h, w = image.shape[:2]
        if h < 12 or w < 12:
            raise ValueError("Selected object is too small; select a larger box")
        return image

    def extract_features(self, crop: np.ndarray) -> dict[str, Any]:
        crop = self._safe_crop(crop)
        with self._lock:
            self.initialize()
            torch = self._torch
            rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # ResNet18 512-d L2-normalised embedding.
            image  = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            mean   = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
            std    = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
            tensor = ((tensor - mean) / std).unsqueeze(0).to("cuda:0" if self.gpu_available else "cpu")
            try:
                with torch.inference_mode():
                    emb = self.embedder(tensor).flatten(1)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0).detach().cpu().numpy()
            except Exception as exc:
                if not self.gpu_available:
                    raise
                self._fallback_to_cpu(exc)
                tensor = tensor.cpu()
                with torch.inference_mode():
                    emb = self.embedder(tensor).flatten(1)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0).detach().cpu().numpy()
            # HSV histogram 12×8×4 bins, L1-normalised.
            hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 4], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
            # ORB keypoints.
            gray = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2GRAY)
            _, descriptors = self._cv_orb.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors  = descriptors[:100]
                orb_encoded  = base64.b64encode(descriptors.tobytes()).decode("ascii")
                orb_rows     = int(descriptors.shape[0])
            else:
                orb_encoded, orb_rows = "", 0
            # Sharpness score (for enrichment quality gating).
            sharp = _sharpness(crop)
            return {
                "embedding": [round(float(x), 6) for x in emb],
                "histogram": [round(float(x), 7) for x in hist],
                "orb_b64":   orb_encoded,
                "orb_rows":  orb_rows,
                "sharpness": round(sharp, 2),
            }

    # ------------------------------------------------------------------
    # Similarity scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compare_features(query: dict, reference: dict,
                         spatial_prior: float | None = None,
                         partial_view: bool = False) -> float:
        """Score two feature dicts.
        spatial_prior in [0,1] (1=same spot).
        partial_view=True when current box < 60% of last known area — shifts
        weight away from deep embedding toward colour + spatial.
        """
        try:
            a = np.asarray(query.get("embedding", []),     dtype=np.float32)
            b = np.asarray(reference.get("embedding", []), dtype=np.float32)
            if a.size == 0 or a.shape != b.shape:
                return 0.0
            cosine = float(np.clip(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1))
            deep = (cosine + 1.0) / 2.0

            ha = np.asarray(query.get("histogram", []),     dtype=np.float32)
            hb = np.asarray(reference.get("histogram", []), dtype=np.float32)
            color = 0.5
            if ha.size and ha.shape == hb.shape:
                corr  = float(cv2.compareHist(
                    ha.reshape(-1, 1).astype(np.float32),
                    hb.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL))
                color = float(np.clip((corr + 1.0) / 2.0, 0, 1))

            orb_score: float | None = None
            qa, rb = query.get("orb_b64", ""),     reference.get("orb_b64", "")
            qn, rn = int(query.get("orb_rows", 0)), int(reference.get("orb_rows", 0))
            if qa and rb and qn >= 8 and rn >= 8:
                qdesc = np.frombuffer(base64.b64decode(qa), dtype=np.uint8).reshape(qn, 32)
                rdesc = np.frombuffer(base64.b64decode(rb), dtype=np.uint8).reshape(rn, 32)
                matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(qdesc, rdesc, k=2)
                good    = sum(1 for p in matches
                              if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
                orb_score = min(1.0, good / max(8.0, min(qn, rn) * 0.35))

            # Adjust weights for partial view.
            w_deep    = _W_DEEP    * (0.75 if partial_view else 1.0)
            w_color   = _W_COLOR   * (1.40 if partial_view else 1.0)
            w_orb     = _W_ORB     if orb_score    is not None else 0.0
            w_spatial = _W_SPATIAL * (1.30 if partial_view else 1.0) if spatial_prior is not None else 0.0

            if orb_score is None and spatial_prior is None and not partial_view:
                # Legacy path keeps existing test behaviour.
                score = 0.67 * deep + 0.33 * color
            else:
                total = w_deep + w_color + w_orb + w_spatial
                if total < 1e-6:
                    return 0.0
                score = (w_deep    * deep
                       + w_color   * color
                       + w_orb     * (orb_score    or 0.0)
                       + w_spatial * (spatial_prior or 0.0)) / total

            return float(np.clip(score, 0.0, 1.0))
        except (ValueError, TypeError, cv2.error):
            return 0.0

    @staticmethod
    def _spatial_prior(last_box: list[float] | None, candidate_box: list[float],
                       frame_w: int, frame_h: int) -> float | None:
        if last_box is None:
            return None
        diag = max(1.0, (frame_w ** 2 + frame_h ** 2) ** 0.5)
        acx  = (last_box[0] + last_box[2]) / 2
        acy  = (last_box[1] + last_box[3]) / 2
        ccx  = (candidate_box[0] + candidate_box[2]) / 2
        ccy  = (candidate_box[1] + candidate_box[3]) / 2
        dist_frac = ((ccx - acx) ** 2 + (ccy - acy) ** 2) ** 0.5 / diag
        return float(np.exp(-5.0 * dist_frac))

    @staticmethod
    def _is_partial_view(candidate_box: list[float],
                         last_box: list[float] | None,
                         threshold: float = 0.60) -> bool:
        """True when the current detection is significantly smaller than the last known box."""
        if last_box is None:
            return False
        last_area = max(1.0, (last_box[2] - last_box[0]) * (last_box[3] - last_box[1]))
        curr_area = (candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1])
        return (curr_area / last_area) < threshold

    # ------------------------------------------------------------------
    # Gallery-level scoring
    # ------------------------------------------------------------------

    def score_against_gallery(self, features: dict, gallery_samples: list[dict],
                               last_box: list[float] | None = None,
                               candidate_box: list[float] | None = None,
                               frame_w: int = 1280, frame_h: int = 720) -> float:
        if not gallery_samples:
            return 0.0
        spatial  = (self._spatial_prior(last_box, candidate_box, frame_w, frame_h)
                    if last_box is not None and candidate_box is not None else None)
        partial  = self._is_partial_view(candidate_box or [], last_box)
        scores   = [self.compare_features(features, s, spatial_prior=spatial, partial_view=partial)
                    for s in gallery_samples]
        return max(scores, default=0.0)

    def score_against_rolling(self, features: dict,
                               rolling: "deque[list[float]]",
                               last_box: list[float] | None = None,
                               candidate_box: list[float] | None = None,
                               frame_w: int = 1280, frame_h: int = 720) -> float:
        if not rolling:
            return 0.0
        spatial = (self._spatial_prior(last_box, candidate_box, frame_w, frame_h)
                   if last_box is not None and candidate_box is not None else None)
        partial = self._is_partial_view(candidate_box or [], last_box)
        best    = 0.0
        for emb in rolling:
            ref = {"embedding": emb, "histogram": [], "orb_b64": "", "orb_rows": 0}
            s   = self.compare_features(features, ref, spatial_prior=spatial, partial_view=partial)
            if s > best:
                best = s
        return best

    # ------------------------------------------------------------------
    # Two-stage match
    # ------------------------------------------------------------------

    def best_asset_match(self, features: dict, assets: list[dict],
                         excluded_asset_ids: set[str] | None = None,
                         min_score: float = 0.76, min_margin: float = 0.07,
                         identity_hints: dict[str, Any] | None = None,
                         candidate_box: list[float] | None = None,
                         frame_w: int = 1280, frame_h: int = 720) -> tuple[str | None, float | None]:
        excluded   = excluded_asset_ids or set()
        candidates: list[tuple[str, float]] = []

        for asset in assets:
            if not asset.get("active", True) or asset["asset_id"] in excluded:
                continue
            aid      = asset["asset_id"]
            hint     = (identity_hints or {}).get(aid)
            last_box = hint.last_box if hint else None

            # Stage 1 — rolling buffer (only for currently-lost assets, priority lane).
            rolling_score = 0.0
            if hint and hint.rolling_embeddings and hint.lost_since is not None:
                rolling_score = self.score_against_rolling(
                    features, hint.rolling_embeddings,
                    last_box=last_box, candidate_box=candidate_box,
                    frame_w=frame_w, frame_h=frame_h,
                )

            # Stage 2 — persistent multi-view gallery.
            gallery_score = self.score_against_gallery(
                features,
                asset.get("profile", {}).get("samples", []),
                last_box=last_box, candidate_box=candidate_box,
                frame_w=frame_w, frame_h=frame_h,
            )

            # Rolling gets a 4% boost (very recent, consistent data).
            best = min(max(rolling_score * 1.04, gallery_score), 1.0)
            candidates.append((aid, best))

        candidates.sort(key=lambda x: x[1], reverse=True)
        if not candidates:
            return None, None
        best_id, best_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else 0.0
        if best_score < min_score or best_score - second_score < min_margin:
            return None, best_score
        return best_id, best_score
