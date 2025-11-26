# KidneyNet Backend API

Flask backend for kidney stone detection predictions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or if using venv:
```bash
source ../venv/bin/activate
pip install flask flask-cors
```

## Run

```bash
python app.py
```

The API will run at `http://localhost:5000`

## Endpoints

- `GET /health` - Health check
- `POST /predict` - Upload image and get prediction
  - Form data with `image` field
  - Returns JSON with prediction, confidence, and probabilities

## Deploy

You can deploy this to:
- Railway: https://railway.app
- Render: https://render.com
- Fly.io: https://fly.io

Make sure to update the frontend's `NEXT_PUBLIC_API_URL` to point to your deployed backend.

