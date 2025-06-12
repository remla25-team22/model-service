# model-service

This repository contains the **model-serving component** for the REMLA project (Team 22).  
It wraps the trained ML model (from `model-training`) in a RESTful API using Flask, enabling prediction requests via HTTP.

---

##  Project Structure

```
model-service/
├── app.py                     # Main Flask app exposing REST endpoints
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition for deployment
├── .github/workflows/         # CI/CD automation
│   └── release.yml            # Docker image build + tag + push
├── .gitignore
└── README.md
```

---

## REST API

The Flask app exposes a REST endpoint to receive user text input and return a sentiment prediction.

- **POST /predict**
  - Request JSON: `{ "text": "the food was amazing" }`
  - Response JSON: `{ "label": "positive" }`

---

## Reuse of Shared Libraries

- Depends on `lib-ml` (installed via versioned Git tag)
  - Handles text preprocessing consistently with training
- `lib-ml` is installed via:
```text
git+https://github.com/remla25-team22/lib-ml.git@v1.0.0
```

---

## Docker Container

- The service is packaged in a Docker container
- Uses multi-stage builds to keep image size small
- Versioned images are pushed to GitHub Container Registry
- URL of the model can be passed via `MODEL_URL` environment variable (if loaded remotely)


