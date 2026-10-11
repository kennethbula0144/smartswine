# Smart Swine FastAPI backend

This backend loads the AI integration bundle from:

`../aitrain/PigAI_Webapp_Integration`

## Install

From the workspace root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

## Run

```powershell
python -m uvicorn backend.app:app --reload --port 8000
```

## Check the setup

- `GET http://127.0.0.1:8000/api/health`
- `GET http://127.0.0.1:8000/api/ai-camera/status`
- `POST http://127.0.0.1:8000/api/ai-camera/process-image`

The image endpoint expects a multipart upload using the field name `image`.

Set `PIG_AI_BUNDLE_DIR` if the AI integration folder is stored somewhere else. The default points to the existing `aitrain/PigAI_Webapp_Integration` folder beside this project.
