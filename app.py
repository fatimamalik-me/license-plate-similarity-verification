import os
import time
from typing import Optional

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import SiameseNet

MODEL_PATH = "best_model.pth"
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 200
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
EMBEDDING_DIM = 128
SIMILARITY_THRESHOLD = 0.50  # Replace with the best validation threshold from V3.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="License Plate Verification",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(37,99,235,.13), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(14,165,233,.11), transparent 28%),
                linear-gradient(135deg, #f8fafc 0%, #edf4ff 48%, #f8fafc 100%);
        }
        .block-container {max-width:1180px; padding-top:2rem; padding-bottom:3rem;}
        #MainMenu, footer, header {visibility:hidden;}
        .hero-card {
            background:rgba(255,255,255,.91); border:1px solid rgba(148,163,184,.30);
            border-radius:24px; padding:2.3rem 2rem; margin-bottom:2rem; text-align:center;
            box-shadow:0 18px 46px rgba(15,23,42,.09); backdrop-filter:blur(10px);
        }
        .hero-badge {
            display:inline-block; color:#1d4ed8; background:#dbeafe; border-radius:999px;
            padding:.35rem .85rem; margin-bottom:.85rem; font-size:.82rem; font-weight:700;
            letter-spacing:.03em;
        }
        .hero-title {color:#0f172a; font-size:2.65rem; font-weight:780; line-height:1.2; margin-bottom:.65rem;}
        .hero-subtitle {color:#64748b; font-size:1.02rem; line-height:1.7; max-width:760px; margin:0 auto;}
        .section-title {color:#0f172a; font-size:1.6rem; font-weight:750; margin-top:.5rem; margin-bottom:.25rem;}
        .section-description {color:#64748b; font-size:.95rem; margin-bottom:1.2rem;}
        .upload-heading {color:#0f172a; font-size:1.35rem; font-weight:720; margin-bottom:.15rem;}
        .upload-description {color:#64748b; font-size:.9rem; margin-bottom:.8rem;}
        [data-testid="stFileUploader"] {
            background:rgba(255,255,255,.92); border:1px solid #dbe4f0; border-radius:16px;
            padding:.75rem; box-shadow:0 8px 24px rgba(15,23,42,.06);
        }
        [data-testid="stFileUploaderDropzone"] {background:#f8fafc; border:1.5px dashed #94a3b8; border-radius:12px;}
        [data-testid="stImage"] img {
            max-height:340px; width:100%; object-fit:contain; border-radius:16px;
            border:1px solid #e2e8f0; background:#fff; box-shadow:0 10px 28px rgba(15,23,42,.08);
        }
        div.stButton > button {
            width:100%; min-height:3.25rem; border:none; border-radius:13px;
            background:linear-gradient(90deg,#2563eb 0%,#0284c7 100%); color:white;
            font-size:1rem; font-weight:720; letter-spacing:.01em;
            box-shadow:0 10px 25px rgba(37,99,235,.26); transition:all .2s ease-in-out;
        }
        div.stButton > button:hover {transform:translateY(-1px); color:white; box-shadow:0 14px 31px rgba(37,99,235,.34);}
        .result-card {
            background:rgba(255,255,255,.94); border:1px solid rgba(148,163,184,.30);
            border-radius:21px; padding:1.65rem; margin-top:1.5rem;
            box-shadow:0 16px 40px rgba(15,23,42,.09);
        }
        .result-heading {color:#0f172a; font-size:1.55rem; font-weight:760; margin-bottom:1rem;}
        [data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0; border-radius:15px; padding:1rem;}
        [data-testid="stMetricLabel"] {color:#64748b;}
        [data-testid="stMetricValue"] {color:#0f172a; font-weight:730;}
        [data-testid="stAlert"] {border-radius:13px;}
        [data-testid="stExpander"] {
            background:rgba(255,255,255,.92); border:1px solid #dbe4f0; border-radius:15px;
            box-shadow:0 8px 24px rgba(15,23,42,.05);
        }
        .custom-footer {text-align:center; color:#94a3b8; font-size:.82rem; margin-top:2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

inference_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def prepare_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    return inference_transform(image).unsqueeze(0).to(DEVICE)


@st.cache_resource
def load_model() -> SiameseNet:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"The trained model file was not found: {MODEL_PATH}")

    model = SiameseNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def read_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(f"'{uploaded_file.name}' is not a valid or supported image.")
        return None


def verify_images(model: SiameseNet, image_a: Image.Image, image_b: Image.Image) -> tuple[float, str]:
    tensor_a = prepare_image(image_a)
    tensor_b = prepare_image(image_b)
    with torch.no_grad():
        embedding_a, embedding_b = model(tensor_a, tensor_b)
        similarity = F.cosine_similarity(embedding_a, embedding_b).item()
    prediction = "Same License Plate" if similarity >= SIMILARITY_THRESHOLD else "Different License Plate"
    return similarity, prediction


if "verification_result" not in st.session_state:
    st.session_state.verification_result = None

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">DEEP METRIC LEARNING APPLICATION</div>
        <div class="hero-title">License Plate Similarity Verification System</div>
        <div class="hero-subtitle">
            Upload two vehicle or license plate images and use the trained Siamese Neural Network
            to determine whether the visible license plates represent the same plate.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Upload Images</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Select two JPG, JPEG, or PNG images. The system will preprocess both images before generating feature embeddings.</div>',
    unsafe_allow_html=True,
)

left_column, right_column = st.columns(2, gap="large")

with left_column:
    st.markdown(
        '<div class="upload-heading">Image A</div><div class="upload-description">Upload the first vehicle or license plate image.</div>',
        unsafe_allow_html=True,
    )
    uploaded_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="image_a", label_visibility="collapsed")
    image_a = read_uploaded_image(uploaded_a)
    if image_a is not None:
        st.image(image_a, caption="Uploaded Image A", use_container_width=True)

with right_column:
    st.markdown(
        '<div class="upload-heading">Image B</div><div class="upload-description">Upload the second vehicle or license plate image.</div>',
        unsafe_allow_html=True,
    )
    uploaded_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="image_b", label_visibility="collapsed")
    image_b = read_uploaded_image(uploaded_b)
    if image_b is not None:
        st.image(image_b, caption="Uploaded Image B", use_container_width=True)

st.write("")
_, button_center, _ = st.columns([1, 1.6, 1])
with button_center:
    verify_clicked = st.button("Verify License Plates", type="primary", use_container_width=True)

if verify_clicked:
    st.session_state.verification_result = None
    if image_a is None or image_b is None:
        st.warning("Please upload two valid images before starting verification.")
    else:
        try:
            model = load_model()
            start_time = time.perf_counter()
            with st.spinner("Generating embeddings and computing similarity..."):
                similarity, prediction = verify_images(model, image_a, image_b)
            processing_time = time.perf_counter() - start_time
            st.session_state.verification_result = {
                "similarity": similarity,
                "prediction": prediction,
                "processing_time": processing_time,
            }
        except FileNotFoundError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error("The trained model could not be loaded. Confirm that the saved weights and SiameseNet architecture in model.py are identical.")
            st.exception(error)
        except Exception as error:
            st.error("Verification could not be completed because of an internal processing error.")
            st.exception(error)

result = st.session_state.verification_result
if result is not None:
    similarity = result["similarity"]
    prediction = result["prediction"]
    processing_time = result["processing_time"]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-heading">Verification Result</div>', unsafe_allow_html=True)
    prediction_column, score_column, time_column = st.columns(3)

    with prediction_column:
        st.metric(label="Prediction", value=prediction)
    with score_column:
        st.metric(label="Cosine Similarity Score", value=f"{similarity:.4f}")
    with time_column:
        st.metric(label="Processing Time", value=f"{processing_time:.3f} sec")

    if prediction == "Same License Plate":
        st.success("The uploaded images are predicted to represent the same license plate.")
    else:
        st.info("The uploaded images are predicted to represent different license plates.")

    st.caption("The cosine similarity score is an embedding similarity measure and should not be interpreted as a calibrated probability.")

    result_text = (
        "LICENSE PLATE SIMILARITY VERIFICATION RESULT\n"
        "--------------------------------------------\n"
        f"Prediction: {prediction}\n"
        f"Cosine similarity score: {similarity:.4f}\n"
        f"Decision threshold: {SIMILARITY_THRESHOLD:.4f}\n"
        f"Processing time: {processing_time:.3f} seconds\n"
    )

    st.download_button(
        label="Download Verification Result",
        data=result_text,
        file_name="license_plate_verification_result.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
with st.expander("Model and System Information", expanded=False):
    information_left, information_right = st.columns(2)
    with information_left:
        st.write("**Backbone network:** ResNet18")
        st.write("**Deep learning framework:** PyTorch")
        st.write(f"**Embedding dimension:** {EMBEDDING_DIM}")
        st.write(f"**Execution device:** {DEVICE}")
    with information_right:
        st.write(f"**Input image size:** {IMAGE_HEIGHT} × {IMAGE_WIDTH}")
        st.write("**Similarity metric:** Cosine similarity")
        st.write(f"**Decision threshold:** {SIMILARITY_THRESHOLD:.4f}")
        st.write("**Application framework:** Streamlit")

st.markdown(
    '<div class="custom-footer">Master\'s Capstone Project — Siamese Neural Network-Based License Plate Similarity Verification</div>',
    unsafe_allow_html=True,
)
