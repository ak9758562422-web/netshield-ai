import io
from pathlib import Path
import gdown, joblib, numpy as np, pandas as pd, streamlit as st, tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

MODEL_ID = "1KoPwcIvWoNFBFpTIf4AEE4yFChZn3gS4"
PREPROCESS_ID = "1I_G8Rl7w5yVkqrG0EwHeziEsiahqfiOs"
MODEL_PATH = Path("/tmp/vgg19_dos_detection.keras")
PREPROCESS_PATH = Path("/tmp/vgg19_dos_preprocessing.joblib")
BATCH_SIZE, MAX_ROWS = 64, 50000

st.set_page_config(page_title="NetShield AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.stApp{background:#07111f;color:#eaf3ff}.block-container{max-width:1200px;padding-top:2rem}
h1,h2,h3{color:#eaf3ff}div[data-testid="stMetric"]{background:#0c1b2d;border:1px solid #1d344c;padding:14px;border-radius:12px}
.stButton>button{background:#52e0cf;color:#06151f;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Downloading and loading the original VGG19 model…")
def load_model():
    if not MODEL_PATH.exists():
        if not gdown.download(id=MODEL_ID, output=str(MODEL_PATH), quiet=False, fuzzy=True):
            raise RuntimeError("Could not download VGG19 model. Check Google Drive link sharing.")
    if not PREPROCESS_PATH.exists():
        if not gdown.download(id=PREPROCESS_ID, output=str(PREPROCESS_PATH), quiet=False, fuzzy=True):
            raise RuntimeError("Could not download preprocessing artifact. Check Google Drive link sharing.")
    artifacts = joblib.load(PREPROCESS_PATH)
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return model, artifacts

def signed_log(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))

def predict_csv(frame, model, artifacts):
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    if "Label" in frame.columns:
        frame = frame.drop(columns=["Label"])
    required = list(artifacts["keep_columns"])
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required features. First missing: {missing[:10]}")
    x = frame[required].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(artifacts["medians"])
    x = signed_log(x.to_numpy())
    x = np.clip(artifacts["minmax"].transform(x), 0, 1)
    x = artifacts["selector"].transform(x)[:, artifacts["order"]]
    n = int(artifacts["n_features"])
    if x.shape[1] < n:
        x = np.hstack([x, np.zeros((len(x), n-x.shape[1]))])
    x = x.astype("float32")
    probabilities = []
    for start in range(0, len(x), BATCH_SIZE):
        images = tf.convert_to_tensor(x[start:start+BATCH_SIZE], dtype=tf.float32)
        images = tf.reshape(images, (-1, 8, 8, 1)) * 255.0
        images = tf.image.resize(images, (64,64), method="nearest")
        images = tf.repeat(images, 3, axis=-1)
        images = preprocess_input(images)
        probabilities.extend(model.predict(images, verbose=0).reshape(-1).astype(float).tolist())
    threshold = float(artifacts["threshold"])
    labels = ["DoS" if p >= threshold else "BENIGN" for p in probabilities]
    result = pd.DataFrame({"row":range(1,len(labels)+1),"prediction":labels,
        "dos_probability":probabilities,"confidence":[max(p,1-p) for p in probabilities]})
    return result, threshold

st.title("🛡️ NetShield AI")
st.subheader("Network Intrusion Detection using VGG19")
st.caption("CIC-IDS2017 · BENIGN vs DoS · Original trained model inference")
with st.expander("Model details"):
    st.write("Architecture: VGG19 fine-tuned binary classifier")
    st.write("Optimizer: Adam · Loss: Binary cross-entropy · Input: 64×64 RGB")
    st.write("Preprocessing: signed log → MinMax → feature selection/order → 8×8 image → 64×64 resize → VGG19 preprocessing")
    st.info("Benchmark figures below are notebook evaluation metrics; uploaded-file results are actual model predictions.")
a,b=st.columns(2)
with a:
    st.metric("VGG19 accuracy", "99.86%")
    st.metric("Precision / Recall / F1", "99.64% / 99.93% / 99.78%")
with b:
    st.markdown("### Upload network traffic")
    file=st.file_uploader("Choose CIC-IDS2017 CSV",type=["csv"])
    st.caption("Maximum 25 MB and 50,000 rows. Keep original feature names.")
if file:
    st.write(f"Selected: {file.name} · {file.size/1048576:.2f} MB")
    if file.size > 25*1024*1024:
        st.error("CSV exceeds 25 MB.")
    elif st.button("Run VGG19 DoS Detection",type="primary",use_container_width=True):
        try:
            df=pd.read_csv(io.BytesIO(file.getvalue()),low_memory=False,encoding="latin1")
            if df.empty: raise ValueError("CSV has no data rows.")
            if len(df)>MAX_ROWS: raise ValueError(f"CSV exceeds {MAX_ROWS:,} rows.")
            with st.spinner("Loading VGG19 and analyzing traffic…"):
                model,artifacts=load_model()
                result,threshold=predict_csv(df,model,artifacts)
            dos=int((result.prediction=="DoS").sum()); benign=len(result)-dos
            st.success(f"Analyzed {len(result):,} flows · Decision threshold {threshold:.4f}")
            c1,c2,c3=st.columns(3)
            c1.metric("Total flows",f"{len(result):,}")
            c2.metric("DoS detected",f"{dos:,}")
            c3.metric("BENIGN traffic",f"{benign:,}")
            st.markdown("### Prediction distribution")
            st.bar_chart(pd.DataFrame({"Flows":[benign,dos]},index=["BENIGN","DoS"]))
            st.markdown("### Predictions (first 1,000 rows)")
            st.dataframe(result.head(1000),use_container_width=True,hide_index=True)
            st.download_button("Download all predictions CSV",result.to_csv(index=False).encode(),
                "netshield_vgg19_predictions.csv","text/csv",use_container_width=True)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("Verify Google Drive sharing and confirm the CSV contains the original CIC-IDS2017 model features.")
st.divider()
st.caption("Research prototype · Actual predictions use the saved VGG19 model. Benchmark metrics are separate.")
