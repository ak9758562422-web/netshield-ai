# NetShield AI — VGG19 DoS Detection

The production frontend is deployed on Vercel at https://netshield-ai-six.vercel.app. It presents the notebook's reported test metrics, VGG19-vs-Random-Forest comparison, classification report, and a CSV upload interface.

The frontend and inference API are separate. The live dashboard will show real predictions only after the Python FastAPI service is deployed and `window.NETSHIELD_API_URL` is configured to its public URL.

- `index.html`: Vercel static dashboard.
- `backend/`: FastAPI inference API, TensorFlow/Keras runtime, actual trained model and preprocessing artifact.
- `render.yaml`: Render Blueprint for the backend service (Standard paid instance; review pricing before confirming).

See `backend/README.md` for backend deployment instructions.
