"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Computer Vision – Environmental Issue Detection
 Detects: Smoke/Air Pollution · Garbage Dumping · Deforestation
=============================================================
"""

import numpy as np
import cv2
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  DETECTION RESULT
# ─────────────────────────────────────────────
@dataclass
class DetectionResult:
    issue_type: str
    confidence: float          # 0–1
    severity: str              # low / medium / high / critical
    bboxes: list = field(default_factory=list)   # [(x, y, w, h), ...]
    mask: Optional[np.ndarray] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    alert_message: str = ""
    annotated_frame: Optional[np.ndarray] = None


# ─────────────────────────────────────────────
#  BASE DETECTOR
# ─────────────────────────────────────────────
class BaseDetector:
    """Abstract base for CV detectors."""

    NAME = "base"
    COLOR = (255, 255, 0)   # BGR

    def detect(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError

    def _severity(self, confidence: float) -> str:
        if confidence >= 0.75: return "critical"
        if confidence >= 0.50: return "high"
        if confidence >= 0.25: return "medium"
        return "low"

    def _annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        out = frame.copy()
        for (x, y, w, h) in result.bboxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), self.COLOR, 2)

        label = f"{result.issue_type} | {result.confidence:.0%} | {result.severity.upper()}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (5, 5), (tw + 15, th + 15), (0, 0, 0), -1)
        cv2.putText(out, label, (10, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR, 2)
        return out


# ─────────────────────────────────────────────
#  SMOKE / AIR POLLUTION DETECTOR
# ─────────────────────────────────────────────
class SmokeDetector(BaseDetector):
    """
    Heuristic smoke detection using:
      - HSV colour space (grey/white hues with low saturation)
      - Laplacian blur metric (smoke reduces sharpness)
      - Intensity variance (smoke creates uniform haze)
    """

    NAME = "Smoke / Air Pollution"
    COLOR = (0, 165, 255)   # orange

    def __init__(self, blur_threshold: float = 100.0, haze_threshold: float = 0.15):
        self.blur_threshold = blur_threshold
        self.haze_threshold = haze_threshold

    def detect(self, frame: np.ndarray) -> DetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Blur detection (low Laplacian variance → haze)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = lap_var < self.blur_threshold

        # ── Colour mask: grey-white smoke hues
        # Smoke: low saturation, medium-to-high value
        smoke_mask = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
        smoke_ratio = smoke_mask.sum() / (255 * smoke_mask.size)

        # ── Local variance (uniformity check)
        local_std = float(gray.std())
        is_haze = local_std < (255 * self.haze_threshold)

        # ── Composite score
        score = 0.0
        if is_blurry:   score += 0.35
        if smoke_ratio > 0.30: score += 0.40
        if is_haze:     score += 0.25
        score = min(score, 1.0)

        # ── Find contours as bounding boxes
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]

        severity = self._severity(score)
        alert = (
            f"⚠️ Smoke/Pollution detected ({score:.0%} confidence). "
            f"Laplacian={lap_var:.1f}, smoke_pixels={smoke_ratio:.1%}"
            if score > 0.1 else ""
        )

        result = DetectionResult(
            issue_type=self.NAME,
            confidence=score,
            severity=severity,
            bboxes=bboxes,
            mask=smoke_mask,
            alert_message=alert,
        )
        result.annotated_frame = self._annotate(frame, result)
        return result


# ─────────────────────────────────────────────
#  GARBAGE / ILLEGAL DUMPING DETECTOR
# ─────────────────────────────────────────────
class GarbageDetector(BaseDetector):
    """
    Detects garbage dumping using:
      - Colour segmentation for common waste colours (brown, dark green, grey)
      - Edge density (cluttered scenes)
      - Texture entropy (irregular waste surfaces)
    """

    NAME = "Garbage / Illegal Dumping"
    COLOR = (0, 0, 255)   # red

    def detect(self, frame: np.ndarray) -> DetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Colour masks for waste materials
        # Browns (cardboard, soil contamination)
        brown_mask = cv2.inRange(hsv, (10, 50, 50), (20, 255, 200))
        # Dark clutter
        dark_mask  = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))

        waste_mask = cv2.bitwise_or(brown_mask, dark_mask)
        waste_ratio = waste_mask.sum() / (255 * waste_mask.size)

        # ── Edge density (clutter creates many edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (255 * edges.size)

        # ── Texture entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))

        # ── Composite
        score = 0.0
        if waste_ratio > 0.25: score += 0.40
        if edge_density > 0.15: score += 0.35
        if entropy > 6.5:       score += 0.25
        score = min(score, 1.0)

        contours, _ = cv2.findContours(waste_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]

        severity = self._severity(score)
        alert = (
            f"🗑️ Garbage dumping detected ({score:.0%} confidence). "
            f"waste_pixels={waste_ratio:.1%}, edge_density={edge_density:.1%}"
            if score > 0.1 else ""
        )

        result = DetectionResult(
            issue_type=self.NAME,
            confidence=score,
            severity=severity,
            bboxes=bboxes,
            mask=waste_mask,
            alert_message=alert,
        )
        result.annotated_frame = self._annotate(frame, result)
        return result


# ─────────────────────────────────────────────
#  DEFORESTATION DETECTOR
# ─────────────────────────────────────────────
class DeforestationDetector(BaseDetector):
    """
    Detects deforestation signatures:
      - Low green-vegetation pixel ratio
      - Large bare soil / brown patches
      - Irregular clearing patterns
    """

    NAME = "Deforestation / Vegetation Loss"
    COLOR = (0, 255, 0)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Healthy green vegetation
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = green_mask.sum() / (255 * green_mask.size)

        # Bare soil / cleared land
        bare_mask = cv2.inRange(hsv, (5, 30, 80), (30, 200, 220))
        bare_ratio = bare_mask.sum() / (255 * bare_mask.size)

        # Score: low green + high bare = deforestation risk
        score = 0.0
        if green_ratio < 0.15: score += 0.50   # very little green
        if bare_ratio > 0.30:  score += 0.50   # lots of bare soil
        score = min(score, 1.0)

        contours, _ = cv2.findContours(bare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]

        severity = self._severity(score)
        alert = (
            f"🌲 Vegetation loss detected ({score:.0%} confidence). "
            f"green={green_ratio:.1%}, bare={bare_ratio:.1%}"
            if score > 0.1 else ""
        )

        result = DetectionResult(
            issue_type=self.NAME,
            confidence=score,
            severity=severity,
            bboxes=bboxes,
            mask=bare_mask,
            alert_message=alert,
        )
        result.annotated_frame = self._annotate(frame, result)
        return result


# ─────────────────────────────────────────────
#  MULTI-ISSUE PIPELINE
# ─────────────────────────────────────────────
class EnvironmentalVisionPipeline:
    """
    Runs all detectors on a frame or video stream
    and returns a unified detection report.
    """

    def __init__(self):
        self.detectors = [
            SmokeDetector(),
            GarbageDetector(),
            DeforestationDetector(),
        ]

    def analyse_frame(self, frame: np.ndarray) -> list[DetectionResult]:
        return [d.detect(frame) for d in self.detectors]

    def analyse_image_path(self, path: str) -> list[DetectionResult]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        return self.analyse_frame(frame)

    def analyse_video(self, video_path: str,
                      sample_every: int = 30,
                      max_frames: int = 100) -> list[dict]:
        """Process video file, sampling every N frames."""
        cap = cv2.VideoCapture(video_path)
        all_results = []
        frame_count = 0
        processed = 0

        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_every == 0:
                results = self.analyse_frame(frame)
                all_results.append({
                    "frame": frame_count,
                    "results": results,
                })
                processed += 1
            frame_count += 1

        cap.release()
        return all_results

    def live_stream(self, camera_id: int = 0) -> None:
        """Real-time detection from webcam."""
        cap = cv2.VideoCapture(camera_id)
        print("[CV] Press 'q' to quit live stream")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.analyse_frame(frame)

            # Overlay results
            y_offset = 30
            for res in results:
                if res.confidence > 0.1:
                    colour = {"low": (0,255,0), "medium": (0,165,255),
                              "high": (0,0,255), "critical": (128,0,128)}.get(res.severity, (255,255,255))
                    label = f"{res.issue_type}: {res.confidence:.0%} [{res.severity}]"
                    cv2.putText(frame, label, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
                    y_offset += 25

            cv2.imshow("Environmental Issue Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def summary_report(self, results: list[DetectionResult]) -> dict:
        """Produce a structured JSON-serialisable report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "detections": [
                {
                    "issue":      r.issue_type,
                    "confidence": round(r.confidence, 3),
                    "severity":   r.severity,
                    "alert":      r.alert_message,
                    "bbox_count": len(r.bboxes),
                }
                for r in results
            ],
            "alerts": [r.alert_message for r in results if r.alert_message],
            "highest_severity": max(
                (r.severity for r in results),
                key=lambda s: ["low","medium","high","critical"].index(s),
                default="low",
            ),
        }


# ── DEMO (synthetic test frame) ──────────────
if __name__ == "__main__":
    print("[CV] Generating synthetic test frame …")

    # Create a 480×640 RGB frame simulating smoke conditions
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200   # grey-ish haze

    # Add some colour patches to simulate garbage
    frame[300:400, 100:300] = [30, 60, 90]   # dark dump
    frame[350:450, 400:600] = [20, 40, 10]   # dark patch

    pipeline = EnvironmentalVisionPipeline()
    results = pipeline.analyse_frame(frame)
    report  = pipeline.summary_report(results)

    print("\n── Detection Report ──")
    for d in report["detections"]:
        print(f"  {d['issue']}: confidence={d['confidence']:.0%}, severity={d['severity']}")
    print(f"Highest severity: {report['highest_severity']}")
