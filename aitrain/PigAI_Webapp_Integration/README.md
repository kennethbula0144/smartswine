# PigAI Webapp Integration Bundle

This is the single folder to export and copy into the existing FastAPI website.

## Bundle contents

```text
PigAI_Webapp_Integration/
├── integration/
│   ├── __init__.py
│   ├── config.py
│   ├── routes.py
│   └── service.py
├── models/
│   ├── pig_model.pt
│   └── pig_detector.pt
├── camera_weight_model.joblib
├── requirements.txt
└── README.md
```

## Install

Copy this entire folder into the root of the existing website. The final website layout should be:

```text
your-website/
├── app.py
├── recognize.py
├── train.py
├── integration/
├── models/
├── camera_weight_model.joblib
└── ...
```

Install the bundle requirements from the website root:

```powershell
pip install -r integration/requirements.txt
```

## Connect the existing FastAPI app

Add this import in `app.py`:

```python
from integration.routes import router as ai_camera_router
```

After creating the FastAPI app, add:

```python
app.include_router(ai_camera_router)
```

Start the website:

```powershell
python -m uvicorn app:app --reload
```

## Endpoints

- `GET /api/ai-camera/status`
- `POST /api/ai-camera/process-image`

Send a camera image using multipart field `image`. The response contains one result per detected pig, including its identity, recognition confidence, bounding box, and estimated weight.

The log is created automatically as `camera_log.csv` in the website root.

## Important

Add your own pig enrollment images to the website's existing dataset folders, retrain the recognition model, and replace `models/pig_model.pt` with the newly trained model. The bundled model is only the starting model from the current project.

Protect the upload endpoint with authentication and upload-size limits before deploying publicly.
