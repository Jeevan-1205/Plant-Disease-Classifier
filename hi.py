import json
import io
from pathlib import Path
from typing import Tuple, Dict, Any
import torch
from torch import nn
from torchvision import models, transforms
from collections import OrderedDict
from PIL import Image
import streamlit as st

# ---- Page config ----
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

# ---- Helpers ----
@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

@st.cache_resource
def load_label_map(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# Improved model loader: accepts a Path-like object and returns (model, idx_to_class)
@st.cache_resource
def load_model(ckpt_path: str) -> Tuple[torch.nn.Module, Dict[int, str]]:
    cp = Path(ckpt_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint to CPU first
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Instantiate base model
    try:
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
    except Exception:
        model = models.resnet152(pretrained=True)

    # Replace classifier (shape must match saved checkpoint)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 512)),
        ('relu', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(512, 39)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.fc = classifier

    # Load state dict safely
    state = checkpoint.get('state_dict', checkpoint)
    # If keys have 'module.' prefix (from DataParallel) remove it
    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '')
        new_state[nk] = v

    model.load_state_dict(new_state)

    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else {}

    model.eval()
    return model, idx_to_class

# Preprocessing
def process_image(pil_img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tfm(pil_img)

# Prediction function
def predict(model: torch.nn.Module, idx_to_class: Dict[int, str], pil_img: Image.Image, topk: int = 5) -> Tuple[list, list, list]:
    tensor = process_image(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.exp(out).squeeze(0)
        top_probs, top_idx = probs.topk(topk)
        top_probs = top_probs.detach().cpu().numpy().tolist()
        top_idx = top_idx.detach().cpu().numpy().tolist()
        top_classes = [idx_to_class.get(i, str(i)) for i in top_idx]
    return top_probs, top_classes, [None]*len(top_classes)

# Optional Grad-CAM (best-effort)
def compute_gradcam(model: torch.nn.Module, pil_img: Image.Image):
    try:
        # Try torchcam if available
        from torchvision.transforms.functional import to_tensor
        from torchcam.methods import SmoothGradCAMpp
        model_cam = model
        cam_extractor = SmoothGradCAMpp(model_cam)
        input_tensor = process_image(pil_img.convert("RGB")).unsqueeze(0)
        out = model_cam(input_tensor)
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        # activation_map is a tensor (H,W) - convert to PIL and return
        amap = activation_map[0].detach().cpu()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap_img = transforms.ToPILImage()(amap.unsqueeze(0))
        return amap_img
    except Exception:
        return None

# ---- Sidebar ----
with st.sidebar:
    st.header("Model & Settings")
    st.markdown(f"**Device:** {DEVICE}")

    # Default values (user may override)
    default_cat_path = r"D:\Plant-diseases-classifier-master\categories.json"
    default_ckpt_path = r"C:\Users\jeevan\Downloads\checkpoint (1).pth"

    cat_path = st.text_input("Label map JSON path", value=default_cat_path)
    ckpt_option = st.radio("Checkpoint source", ("Use default path on disk", "Upload .pth file"))

    uploaded_ckpt = None
    if ckpt_option == "Upload .pth file":
        uploaded_ckpt = st.file_uploader("Upload checkpoint (.pth)", type=["pth"]) 

    topk = st.slider("Top-K results", 1, 10, 5)
    show_gradcam = st.checkbox("Compute Grad-CAM (slow)", value=False)
    show_raw_logits = st.checkbox("Show raw logits (debug)", value=False)
    confidence_threshold = st.slider("Confidence threshold to flag as 'high'", 0.0, 1.0, 0.5)

# ---- Main UI ----
st.title("🌿 Plant Disease Classifier 🌿")

# Load label map
cat_to_name = load_label_map(cat_path)

# Load model (either uploaded or default path)
model = None
idx_to_class = {}
load_error = None

if uploaded_ckpt:
    # Save to temp and load
    tmp_path = Path(st.secrets.get("_TMP_DIR", ".")) / "uploaded_ckpt.pth"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_ckpt.read())
    try:
        model, idx_to_class = load_model(str(tmp_path))
    except Exception as e:
        load_error = str(e)
else:
    try:
        model, idx_to_class = load_model(default_ckpt_path)
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(f"Failed to load model: {load_error}")
    st.stop()

model = model.to(DEVICE)

# Layout: two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    st.markdown("Upload an image, pick a sample, or use your camera.")

    uploaded = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"]) 
    sample_dir = st.text_input("Optional: folder with sample images (local path)", value="")

    cam_img = None
    use_camera = st.checkbox("Use camera input (experimental)")
    if use_camera:
        cam_img = st.camera_input("Take a photo")

    sample_files = []
    if sample_dir:
        try:
            p = Path(sample_dir)
            sample_files = [str(x) for x in p.glob("*.jpg")] + [str(x) for x in p.glob("*.png")]
            if sample_files:
                choice = st.selectbox("Pick a sample image", options=["--none--"] + sample_files)
                if choice and choice != "--none--":
                    uploaded = open(choice, "rb")
        except Exception:
            st.warning("Could not read sample directory")

    input_img = None
    if cam_img:
        input_img = Image.open(io.BytesIO(cam_img.getvalue()))
    elif uploaded:
        if hasattr(uploaded, "read"):
            input_img = Image.open(io.BytesIO(uploaded.read()))
        else:
            # streamlit sometimes returns an UploadedFile that works with PIL directly
            try:
                input_img = Image.open(uploaded)
            except Exception:
                input_img = None

    if input_img is None:
        st.info("Upload or capture an image to get predictions.")
    else:
        st.image(input_img, caption="Input image", use_column_width=True)

with col2:
    st.subheader("Model / Info")
    with st.expander("Model details"):
        st.write(f"ResNet-152 fine-tuned head — device: {DEVICE}")
        st.write(f"Number of classes (label map): {len(cat_to_name)}")
        st.write("Index to class (sample):")
        if idx_to_class:
            # show first 10
            preview = {k: idx_to_class[k] for k in list(idx_to_class)[:10]}
            st.json(preview)
        else:
            st.write("(no idx_to_class available in checkpoint)")

    with st.expander("Class list (human-readable)"):
        if cat_to_name:
            st.dataframe([(k, cat_to_name[k]) for k in sorted(cat_to_name.keys())], use_container_width=True)
        else:
            st.write("No label map loaded — provide a valid categories.json path in the sidebar.")

    with st.expander("Help / Instructions"):
        st.markdown("""
        - Upload a leaf image or use the camera.
        - Adjust Top-K on the sidebar.
        - Optional: enable Grad-CAM to get a saliency map (may require extra packages).
        - Use the confidence threshold to mark high-confidence predictions.
        """)

# ---- PDF generator function (in-memory) ----
def make_predictions_pdf_bytes(results: list, input_image: Image.Image = None) -> bytes:
    """
    results: list of dicts with keys 'class_id', 'label', 'probability'
    input_image: PIL.Image (optional) to include in the PDF
    returns: PDF file bytes
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from datetime import datetime

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("🌿 Plant Disease Classifier — Predictions Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Include the input image if provided (resize to fit)
    if input_image is not None:
        try:
            img_buf = io.BytesIO()
            # Convert to RGB and save as JPEG to buffer
            input_image.convert("RGB").save(img_buf, format="JPEG")
            img_buf.seek(0)
            # Use reportlab Image and constrain width to 4 inches
            rl_img = RLImage(img_buf, width=4*inch, height=None)
            story.append(rl_img)
            story.append(Spacer(1, 12))
        except Exception:
            # If embedding fails, continue without the image
            pass

    # Summary (top prediction)
    if results:
        top = results[0]
        summary_text = f"<b>Top prediction:</b> {top['label']} ({top['class_id']}) — {top['probability']*100:.2f}%"
    else:
        summary_text = "No predictions available."
    story.append(Paragraph(summary_text, styles["Heading2"]))
    story.append(Spacer(1, 12))

    # ---- Improved table: use Paragraphs so long text wraps nicely ----
    # Create a small custom paragraph style for table cells
    cell_style = ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        alignment=TA_LEFT,
        fontSize=9,
        leading=11,
    )
    header_style = ParagraphStyle(
        name="TableHeader",
        parent=styles["Heading4"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=12,
    )

    data = [
        [
            Paragraph("<b>Rank</b>", header_style),
            Paragraph("<b>Class ID</b>", header_style),
            Paragraph("<b>Label</b>", header_style),
            Paragraph("<b>Probability (%)</b>", header_style)
        ]
    ]

    for i, r in enumerate(results, start=1):
        data.append([
            Paragraph(str(i), cell_style),
            Paragraph(str(r.get("class_id", "")), cell_style),
            Paragraph(str(r.get("label", "")), cell_style),
            Paragraph(f"{r.get('probability',0)*100:.2f}", cell_style),
        ])

    # Increase width for Class ID and Label columns and allow wrapping
    table = Table(
        data,
        colWidths=[40, 180, 230, 90],  # adjust these if you need more room
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9f2d9")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (0,0), (0,-1), "CENTER"),         # Rank col centered
        ("ALIGN", (1,0), (2,-1), "LEFT"),           # Class ID & Label left-aligned
        ("ALIGN", (3,0), (3,-1), "CENTER"),         # Probability centered
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    # ---- end table ----

    # Footer / metadata
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Tip: Use high-resolution well-lit images for best results.", styles["Italic"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---- Run prediction and show results ----
if 'input_img' in locals() and input_img is not None:
    try:
        with st.spinner("Analyzing image..."):
            probs, classes, _ = predict(model, idx_to_class, input_img, topk=topk)

        # Build results table for display & download
        results = []
        for p, c in zip(probs, classes):
            name = cat_to_name.get(c, c)
            results.append({"class_id": c, "label": name, "probability": float(p)})

        st.subheader("Predictions")
        # Show top result as a metric
        best = results[0]
        st.metric(label="Top prediction", value=f"{best['label']} ({best['class_id']})", delta=f"{best['probability']*100:.2f}%")

        # Bar chart and table
        st.write("## Top-K probabilities")
        # Use Streamlit's built-in charting for simplicity
        chart_df = {r['label']: r['probability'] for r in results}
        st.bar_chart(list(chart_df.values()), height=240)
        st.table(results)

        # Progress + badge
        st.progress(min(1.0, float(results[0]['probability'])))
        if results[0]['probability'] >= confidence_threshold:
            st.success(f"High-confidence prediction: {results[0]['label']} ({results[0]['probability']*100:.1f}%)")
        else:
            st.info(f"Low confidence: {results[0]['probability']*100:.1f}%. Consider another photo or higher-resolution image.")

        # Optionally compute Grad-CAM
        if show_gradcam:
            with st.spinner('Computing Grad-CAM...'):
                amap_img = compute_gradcam(model, input_img)
                if amap_img is not None:
                    st.image(amap_img, caption="Grad-CAM / Saliency map", use_column_width=True)
                else:
                    st.warning("Grad-CAM unavailable (missing dependency or failed to compute). Try installing 'torchcam'.")

        # ----- Generate PDF bytes and provide download button -----
        try:
            pdf_bytes = make_predictions_pdf_bytes(results, input_image=input_img)
            st.download_button(
                label="Download predictions (PDF)",
                data=pdf_bytes,
                file_name="predictions.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Failed to generate PDF (is 'reportlab' installed?): {e}")
            # Fallback: keep JSON download so user still has something
            out_json = json.dumps({"predictions": results}, indent=2)
            st.download_button("Download results (JSON)", data=out_json, file_name="predictions.json", mime="application/json")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer / credits
st.markdown("---")
st.caption("Built with ❤️ — ResNet-152 backbone. Tip: use high-resolution, well-lit images for best results.")
