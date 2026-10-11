from __future__ import annotations

import os
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT_MODEL = WEBAPP_DIR / "camera_weight_model.joblib"
DEFAULT_LOG_FILE = WEBAPP_DIR / "camera_log.csv"

MODEL_DIR = Path(os.getenv("PIG_MODEL_DIR", WEBAPP_DIR / "models"))
DEFAULT_IDENTITY_DATASET = WEBAPP_DIR.parents[3] / "pigidentification"
IDENTITY_DATASET_DIR = Path(os.getenv("PIG_IDENTITY_DATASET_DIR", str(DEFAULT_IDENTITY_DATASET)))
DEFAULT_BEHAVIOR_DATASET = WEBAPP_DIR.parents[3] / "-100k-OPB-Dataset-"
BEHAVIOR_DATASET_DIR = Path(os.getenv("PIG_BEHAVIOR_DATASET_DIR", str(DEFAULT_BEHAVIOR_DATASET)))
DEFAULT_WEIGHT_CAPTURE_DIR = WEBAPP_DIR.parents[3] / "4pigsweightdatasets" / "New folder (2)"
WEIGHT_CAPTURE_DIR = Path(os.getenv("PIG_WEIGHT_CAPTURE_DIR", str(DEFAULT_WEIGHT_CAPTURE_DIR)))
WEIGHT_CAPTURE_CAMERA_HEIGHT_M = 2.0
IDENTITY_CLASSES = ("Pig_1", "Pig_2", "Pig_3", "Pig_4")
IDENTITY_WEIGHTS_KG = {
	"Pig_1": 15.67,
	"Pig_2": 13.30,
	"Pig_3": 10.64,
	"Pig_4": 14.20,
}
WEIGHT_MODEL_FILE = Path(os.getenv("PIG_WEIGHT_MODEL_FILE", DEFAULT_WEIGHT_MODEL))
BEHAVIOR_MODEL_FILE = Path(os.getenv("PIG_BEHAVIOR_MODEL_FILE", str(MODEL_DIR / "behavior_model.pt")))
WEIGHT_CALIBRATION_FILE = WEIGHT_MODEL_FILE.with_name("weight_calibration.json")
CAMERA_LOG_FILE = Path(os.getenv("PIG_CAMERA_LOG_FILE", DEFAULT_LOG_FILE))
DETECTOR_CONFIDENCE = float(os.getenv("PIG_DETECTOR_CONFIDENCE", "0.35"))
RECOGNITION_THRESHOLD = float(os.getenv("PIG_RECOGNITION_THRESHOLD", "0.85"))
RECOGNITION_MARGIN = float(os.getenv("PIG_RECOGNITION_MARGIN", "0.20"))
