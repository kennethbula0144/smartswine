from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from torch import nn
from ultralytics import YOLO

from .config import (
    DETECTOR_CONFIDENCE,
    BEHAVIOR_MODEL_FILE,
    IDENTITY_CLASSES,
    MODEL_DIR,
    RECOGNITION_MARGIN,
    RECOGNITION_THRESHOLD,
    WEIGHT_CALIBRATION_FILE,
    WEIGHT_MODEL_FILE,
)

try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None
    transforms = None


class SimplePigClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images):
        features = self.model(images)
        return self.classifier(torch.flatten(features, 1))


class PigCameraService:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.previous_boxes: list[dict[str, Any]] = []
        self.recognition_model, self.classes = self._load_recognition_model()
        self.behavior_model, self.behavior_classes = self._load_behavior_model()
        self.weight_model = joblib.load(WEIGHT_MODEL_FILE) if WEIGHT_MODEL_FILE.exists() else None
        self.weight_calibration = {"slope": 1.0, "intercept": 0.0}
        if WEIGHT_CALIBRATION_FILE.exists():
            try:
                self.weight_calibration.update(json.loads(WEIGHT_CALIBRATION_FILE.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        detector_file = MODEL_DIR / "pig_detector.pt"
        self.detector = YOLO(str(detector_file)) if detector_file.exists() else None

    def _load_recognition_model(self):
        model_file = MODEL_DIR / "pig_model.pt"
        if torch is None or not model_file.exists():
            return None, []
        checkpoint = torch.load(model_file, map_location="cpu")
        classes = checkpoint["classes"]
        if tuple(classes) != IDENTITY_CLASSES:
            return None, []
        model = SimplePigClassifier(num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, classes

    def _load_behavior_model(self):
        if torch is None or not BEHAVIOR_MODEL_FILE.exists():
            return None, []
        checkpoint = torch.load(BEHAVIOR_MODEL_FILE, map_location="cpu")
        classes = checkpoint.get("classes", [])
        if len(classes) < 2:
            return None, []
        model = SimplePigClassifier(num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, classes

    @staticmethod
    def _weight_features(image: Image.Image) -> np.ndarray:
        pixels = np.asarray(image.convert("RGB"))
        return np.asarray([pixels.shape[0] * pixels.shape[1], pixels.mean(axis=(0, 1)).mean()], dtype=float)

    def _recognize(self, crop: Image.Image) -> dict[str, Any]:
        if self.recognition_model is None or transforms is None:
            return {"pig_id": "UNKNOWN", "confidence": 0.0, "known": False}
        tensor = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])(
            crop.convert("RGB")
        ).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.recognition_model(tensor), dim=1)[0]
            values, indices = torch.sort(probabilities, descending=True)
        confidence = float(values[0].item())
        second = float(values[1].item()) if len(values) > 1 else 0.0
        known = confidence >= RECOGNITION_THRESHOLD and confidence - second >= RECOGNITION_MARGIN
        class_name = self.classes[int(indices[0].item())] if known else "UNKNOWN"
        return {"pig_id": class_name, "confidence": confidence * 100, "known": known}

    def _classify_behavior(self, crop: Image.Image) -> dict[str, Any] | None:
        if self.behavior_model is None or transforms is None:
            return None
        tensor = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])(
            crop.convert("RGB")
        ).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.behavior_model(tensor), dim=1)[0]
            confidence, index = torch.max(probabilities, dim=0)
        return {
            "label": self.behavior_classes[int(index.item())],
            "confidence": float(confidence.item()) * 100,
        }

    def _detect(self, image: Image.Image) -> list[dict[str, Any]]:
        if self.detector is None:
            return []
        image_array = np.asarray(image.convert("RGB"))
        results = self.detector.predict(source=image_array, conf=DETECTOR_CONFIDENCE, verbose=False)
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                x1, y1, x2, y2 = [round(value, 2) for value in box]
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": round(confidence * 100, 2)})
        return boxes

    @staticmethod
    def _movement_indicator(box: dict[str, Any], previous_boxes: list[dict[str, Any]], image: Image.Image) -> str:
        if not previous_boxes:
            return "New observation"

        current_center = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        current_width = max(box["x2"] - box["x1"], 1)
        nearest_distance = min(
            (
                ((current_center[0] - ((old["x1"] + old["x2"]) / 2)) ** 2
                 + (current_center[1] - ((old["y1"] + old["y2"]) / 2)) ** 2) ** 0.5
                for old in previous_boxes
            ),
            default=0,
        )
        normalized_motion = nearest_distance / current_width
        if normalized_motion < 0.15:
            return "Stationary"
        if normalized_motion < 0.6:
            return "Moving"
        return "Highly active"

    def process_image(self, image_bytes: bytes) -> dict[str, Any]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        boxes = self._detect(image)
        previous_boxes = self.previous_boxes
        self.previous_boxes = boxes
        pigs = []
        for index, box in enumerate(boxes, start=1):
            coordinates = (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
            crop = image.crop(coordinates)
            identity = self._recognize(crop)
            behavior = self._classify_behavior(crop)
            if self.weight_model is None:
                estimated_weight = None
            else:
                estimated_weight = float(self.weight_model.predict(self._weight_features(crop).reshape(1, -1))[0])
                estimated_weight = (
                    estimated_weight * self.weight_calibration["slope"]
                    + self.weight_calibration["intercept"]
                )
            pig = {
                "pig_id": identity["pig_id"] if identity["known"] else f"UNKNOWN-{index}",
                "recognition_confidence": round(identity["confidence"], 2),
                "estimated_weight_kg": round(estimated_weight, 2) if estimated_weight is not None else None,
                "behavior_indicator": behavior["label"] if behavior else self._movement_indicator(box, previous_boxes, image),
                "behavior_confidence": round(behavior["confidence"], 2) if behavior else None,
                "behavior_source": "OPB behavior model" if behavior else "Movement heuristic",
                "box": box,
            }
            pigs.append(pig)
            self._append_log(pig)
        return {"status": "ok", "pigs": pigs, "detected_count": len(pigs)}

    def _append_log(self, pig: dict[str, Any]) -> None:
        import pandas as pd

        row = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), **pig}])
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(self.log_file, mode="a", header=not self.log_file.exists(), index=False)
