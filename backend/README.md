# NetShield AI — VGG19 inference service

This Docker service loads the actual trained `vgg19_dos_detection.keras` model and matching `vgg19_dos_preprocessing.joblib` artifact. The model archive reports Keras 3.13.2, so the container uses Python 3.11, TensorFlow CPU 2.20.0 and Keras 3.15.1 (Keras 3-compatible runtime family).

## Deploy backend on Render

1. Extract `netshield-ai-deployment.zip` and push the `netshield-ai` folder to a **private GitHub repository**. Keep the trained model artifacts private.
2. In Render, choose **New → Blueprint** and connect that repository. Render reads the root `render.yaml` and creates the inference service.
3. The blueprint uses the **Standard** instance (2 GB RAM) because TensorFlow + VGG19 may exceed the memory available on smaller instances. This is a paid service; review the current price in Render before confirming creation.
4. After deployment, open `https://YOUR-SERVICE.onrender.com/health`. It must return `status: ok` and the saved decision threshold.
5. Copy the service URL. In `index.html`, add `<script>window.NETSHIELD_API_URL = "https://YOUR-SERVICE.onrender.com";</script>` before the dashboard script, then redeploy the static frontend to Vercel. The dashboard reads this browser-side setting; a normal Vercel environment variable is not automatically exposed to static HTML.
6. Ensure the backend `CORS_ORIGINS` includes `https://netshield-ai-six.vercel.app`.

## API

- `GET /health` — model readiness and decision threshold.
- `POST /predict` — multipart form upload with field name `file`; accepts CSV up to 25 MB and 50,000 rows.

Input CSV must contain the CIC-IDS2017 feature headers retained by the training pipeline. `Label` is optional and ignored. The backend does not train the model; it performs inference with the saved model and preprocessing artifacts.

## Local Docker test

```bash
docker build -t netshield-ai-api ./backend
docker run --rm -p 8000:8000 \
  -e CORS_ORIGINS="https://netshield-ai-six.vercel.app" \
  netshield-ai-api
```

Then open `http://localhost:8000/health` and `http://localhost:8000/docs`.
