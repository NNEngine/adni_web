import streamlit as st
import numpy as np
import nibabel as nib
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from scipy.ndimage import zoom

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "model/cnn3d_model.onnx"
SAMPLES_DIR = "samples"

# CHANGE THESE IF NEEDED
TARGET_SHAPE = (128, 128, 128)

CLASS_NAMES = {
    0: "Stable MCI",
    1: "MCI Conversion to Alzheimer's Disease"
}

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ADNI MRI Conversion Predictor",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:25px;
}

.result-box {
    padding:20px;
    border-radius:15px;
    background-color:#1e1e1e;
    border:1px solid #444;
}

.metric-card {
    padding:15px;
    border-radius:10px;
    background:#262730;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='main-title'>🧠 ADNI MRI Conversion Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>3D CNN based MCI Conversion Prediction using ADNI MRI Volumes</div>",
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Model Information")

    st.info("""
    **Dataset:** ADNI

    **Architecture:** 3D CNN

    **Input Size:** 128 × 128 × 128

    **Task:** Predict MCI Conversion

    **Framework:** ONNX Runtime
    """)

    st.divider()

    st.write("### Class Labels")

    st.success("0 → Stable MCI")

    st.error("1 → MCI Conversion")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    return session


try:
    session = load_model()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

except Exception as e:

    st.error(f"Model Loading Failed\n\n{e}")
    st.stop()

# =====================================================
# HELPERS
# =====================================================

def normalize(volume):

    volume = volume.astype(np.float32)

    min_val = np.min(volume)
    max_val = np.max(volume)

    volume = (
        volume - min_val
    ) / (
        max_val - min_val + 1e-8
    )

    return volume


def resize_volume(volume):

    factors = (
        TARGET_SHAPE[0] / volume.shape[0],
        TARGET_SHAPE[1] / volume.shape[1],
        TARGET_SHAPE[2] / volume.shape[2]
    )

    volume = zoom(
        volume,
        factors,
        order=1
    )

    return volume


def preprocess(volume):

    volume = normalize(volume)

    volume = resize_volume(volume)

    # Shape becomes:
    # (1,1,128,128,128)

    volume = np.expand_dims(volume, axis=0)
    volume = np.expand_dims(volume, axis=0)

    volume = volume.astype(np.float32)

    return volume


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(volume):

    x = preprocess(volume)

    output = session.run(
        [output_name],
        {input_name: x}
    )[0]

    logit = float(output.flatten()[0])

    probability = sigmoid(logit)

    prediction = int(probability > 0.5)

    return prediction, probability


def load_nifti(path):

    img = nib.load(path)

    volume = img.get_fdata()

    return volume


# =====================================================
# SAMPLE MRI SECTION
# =====================================================

st.header("📁 Sample MRI Scans")

sample_files = []

sample_path = Path(SAMPLES_DIR)

if sample_path.exists():

    for f in sample_path.iterdir():

        if (
            str(f).endswith(".nii")
            or
            str(f).endswith(".nii.gz")
        ):
            sample_files.append(f)

selected_sample = None

if len(sample_files) > 0:

    selected_sample = st.selectbox(
        "Choose Sample Scan",
        sample_files,
        format_func=lambda x: x.name
    )

else:

    st.warning(
        "No sample scans found in /samples folder"
    )

# =====================================================
# UPLOAD SECTION
# =====================================================

st.header("📤 Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Upload .nii or .nii.gz",
    type=["nii", "gz"]
)

# =====================================================
# LOAD VOLUME
# =====================================================

volume = None

if uploaded_file is not None:

    filename = uploaded_file.name.lower()

    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"

    elif filename.endswith(".nii"):
        suffix = ".nii"

    else:
        st.error(
            "Unsupported file format. Please upload .nii or .nii.gz"
        )
        st.stop()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())

        temp_path = tmp.name

    volume = load_nifti(temp_path)

elif selected_sample is not None:

    volume = load_nifti(str(selected_sample))

# =====================================================
# VISUALIZATION
# =====================================================

if volume is not None:

    st.divider()

    st.header("🧬 MRI Visualization")

    shape_x, shape_y, shape_z = volume.shape

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Width", shape_x)

    with col2:
        st.metric("Height", shape_y)

    with col3:
        st.metric("Depth", shape_z)

    st.write("### Slice Navigator")

    slice_idx = st.slider(
        "Axial Slice",
        0,
        shape_z - 1,
        shape_z // 2
    )

    axial = volume[:, :, slice_idx]

    coronal = volume[:, shape_y // 2, :]

    sagittal = volume[shape_x // 2, :, :]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("Axial")

        fig, ax = plt.subplots()

        ax.imshow(
            axial.T,
            cmap="gray",
            origin="lower"
        )

        ax.axis("off")

        st.pyplot(fig)

    with c2:

        st.subheader("Coronal")

        fig, ax = plt.subplots()

        ax.imshow(
            coronal.T,
            cmap="gray",
            origin="lower"
        )

        ax.axis("off")

        st.pyplot(fig)

    with c3:

        st.subheader("Sagittal")

        fig, ax = plt.subplots()

        ax.imshow(
            sagittal.T,
            cmap="gray",
            origin="lower"
        )

        ax.axis("off")

        st.pyplot(fig)

    # =================================================
    # MRI STATISTICS
    # =================================================

    st.divider()

    st.header("📊 MRI Statistics")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Min",
        f"{np.min(volume):.2f}"
    )

    s2.metric(
        "Max",
        f"{np.max(volume):.2f}"
    )

    s3.metric(
        "Mean",
        f"{np.mean(volume):.2f}"
    )

    s4.metric(
        "Std",
        f"{np.std(volume):.2f}"
    )

    # =================================================
    # PREDICTION
    # =================================================

    st.divider()

    st.header("🚀 Prediction")

    if st.button(
        "Predict MCI Conversion",
        use_container_width=True
    ):

        with st.spinner(
            "Running 3D CNN inference..."
        ):

            prediction, confidence = predict(volume)

        st.divider()

        st.subheader("Prediction Result")

        label = CLASS_NAMES[prediction]

        if prediction == 1:

            st.error(label)

        else:

            st.success(label)

        st.write(
            f"### Confidence: {confidence*100:.2f}%"
        )

        st.progress(float(confidence))

        if confidence >= 0.8:

            st.warning(
                "High confidence prediction."
            )

        elif confidence >= 0.6:

            st.info(
                "Moderate confidence prediction."
            )

        else:

            st.info(
                "Low confidence prediction."
            )

else:

    st.info(
        "Upload a scan or choose a sample MRI."
    )
