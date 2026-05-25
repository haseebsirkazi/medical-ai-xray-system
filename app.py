import streamlit as st
import numpy as np
import cv2
from PIL import Image
from predict import predict, get_heatmap
from report import generate_pdf
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Medical Imaging System",
    layout="wide",
    page_icon="🩻"
)

# ---------------- STYLING ----------------
st.markdown(
    """
    <style>

    .main {
        background-color: #0b0f19;
    }

    .hero {
        text-align: center;
        padding: 10px;
    }

    .title {
        font-size: 48px;
        font-weight: 800;
        color: #00e5ff;
    }

    .subtitle {
        font-size: 18px;
        color: #a0a0a0;
        margin-top: -10px;
    }

    .card {
        background: #121826;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
    }

    .card-title {
        font-size: 18px;
        font-weight: bold;
        color: #00e5ff;
    }

    .card-text {
        font-size: 13px;
        color: #cfcfcf;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="title">🩻 AI Medical Imaging System</div>
    <div class="subtitle">Chest X-ray Diagnosis with Explainable AI (Grad-CAM)</div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🧠 AI Diagnosis</div>
        <div class="card-text">Detect Pneumonia using Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🔥 Explainable AI</div>
        <div class="card-text">Heatmap shows AI attention regions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">📄 Medical Report</div>
        <div class="card-text">Download AI-generated PDF report</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    # ---------------- PREDICTION ----------------
    probs, tensor = predict(image)

    normal = float(probs[0][0])
    pneumonia = float(probs[0][1])

    if pneumonia > normal:
        class_idx = 1
        result_text = "⚠ Pneumonia Detected"
    else:
        class_idx = 0
        result_text = "✔ Normal"

    st.divider()

    # ---------------- HEATMAP ----------------
    cam = get_heatmap(tensor, class_idx)

    img = np.array(image.resize((224, 224)).convert("RGB")).astype("float32")

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.resize(heatmap, (224, 224)).astype("float32")

    overlay = cv2.addWeighted(img, 0.75, heatmap, 0.25, 0)
    overlay = np.uint8(overlay)

    # ---------------- 3 COLUMN COMPARISON ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🩻 Original X-ray")
        st.image(img.astype("uint8"), use_column_width=True)

    with col2:
        st.markdown("### 🧠 Diagnosis")

        if class_idx == 1:
            st.error(result_text)
        else:
            st.success(result_text)

        st.write("**Confidence Scores**")
        st.write(f"🟢 Normal: {normal*100:.2f}%")
        st.write(f"🔴 Pneumonia: {pneumonia*100:.2f}%")

        st.progress(max(normal, pneumonia))

    with col3:
        st.markdown("### 🔥 AI Heatmap")
        st.image(overlay, use_column_width=True)

    # ---------------- PDF REPORT ----------------
    st.divider()

    st.markdown("### 📄 Generate Medical Report")

    if st.button("📥 Download PDF Report"):

        file_path = tempfile.mktemp(suffix=".pdf")

        generate_pdf(
            file_path,
            normal,
            pneumonia,
            result_text
        )

        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇ Download Report",
                data=f,
                file_name="AI_Medical_Report.pdf",
                mime="application/pdf"
            )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🧠 AI Medical Imaging System | Built with PyTorch + Streamlit")