import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tempfile

import streamlit as st

from src.inference import Buzy_inference

st.set_page_config(page_title="Buzy AI", page_icon="🧠", layout="wide")

st.markdown("# Buzy AI")
st.markdown("### AI-Powered Document Intelligence for African Organizations")
st.markdown(
    "Upload documents, images, and meeting audio — ask a business question — "
    "and get a structured, source-attributed analysis powered by **Gemma 4**."
)

with st.expander("Upload files", expanded=True):
    cols = st.columns(3)
    with cols[0]:
        image_files = st.file_uploader(
            "Images (receipts, contracts, dashboards)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
        )
    with cols[1]:
        document_files = st.file_uploader(
            "Documents (PDF, DOCX, TXT)",
            type=[".pdf", ".docx", ".txt"],
            accept_multiple_files=True,
        )
    with cols[2]:
        audio_files = st.file_uploader(
            "Audio (meeting recordings)",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True,
        )

question = st.text_area(
    "Business Question",
    placeholder="e.g. What are the biggest risks in this contract? Which supplier should we prioritize?",
    height=100,
)

if image_files:
    st.markdown("#### Uploaded Images")
    cols = st.columns(min(len(image_files), 4))
    for i, f in enumerate(image_files):
        cols[i % 4].image(f, width="stretch")

st.markdown("#### Quick examples")
ec1, ec2, ec3 = st.columns(3)
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_assets")

q_invoice = "Summarize this invoice and flag any anomalies."
q_contract = "What are the key obligations and risks in this contract?"
q_meeting = "Summarize the key decisions and action items from this meeting."

with ec1:
    if st.button("Analyze Invoice", width="stretch"):
        st.session_state.example_question = q_invoice
        st.rerun()
with ec2:
    if st.button("Review Contract", width="stretch"):
        st.session_state.example_question = q_contract
        st.rerun()
with ec3:
    if st.button("Summarize Meeting", width="stretch"):
        st.session_state.example_question = q_meeting
        st.rerun()

if st.session_state.get("example_question") and not question:
    question = st.session_state.example_question

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("Analyze", type="primary", width="stretch")

if analyze:
    temp_paths = []

    def _save(files):
        paths = []
        if files:
            for f in files:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getvalue())
                    paths.append(tmp.name)
        return paths

    img_paths = _save(image_files)
    doc_paths = _save(document_files)
    aud_paths = _save(audio_files)
    temp_paths = img_paths + doc_paths + aud_paths

    progress_bar = st.progress(0, text="Initializing...")

    def _progress(v, desc=""):
        progress_bar.progress(v, text=desc)

    try:
        report, images = Buzy_inference(
            img_paths, doc_paths, aud_paths, question,
            progress=_progress,
        )
        st.session_state.report = report
        st.session_state.analysis_images = images
    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    st.rerun()

if "report" in st.session_state:
    st.markdown(st.session_state.report)
    if st.session_state.get("analysis_images"):
        st.subheader("Extracted Images")
        for img in st.session_state.analysis_images:
            st.image(img)
