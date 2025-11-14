import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.model import build_model
from src.data import get_transforms


@st.cache_resource
def load_checkpoint(checkpoint_path):
    device = torch.device('cpu')
    try:
        data = torch.load(checkpoint_path, map_location=device)
        classes = data.get('classes', [])
        model_name = data.get('model_name', 'custom_cnn')
        img_size = data.get('img_size', 224)

        model = build_model(num_classes=len(classes), model_name=model_name, pretrained=False)
        model.load_state_dict(data['model_state_dict'])
        model.to(device).eval()
        return model, classes, model_name, img_size, device
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None, None, None, None, None


def predict_image(model, image_pil, classes, img_size, device, topk=5):
    transform = get_transforms(img_size=img_size, train=False)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        topk_probs, topk_indices = torch.topk(probs, k=min(topk, len(classes)))

    results = []
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        results.append((classes[idx], prob))
    return results


def main():
    st.set_page_config(page_title="🌿 Leaf Disease Detection", page_icon="🌿", layout="wide")
    st.title("🌿 Leaf Disease Detection")
    st.markdown("**Identify plant diseases using AI-powered image classification**")
    st.markdown("---")

    st.sidebar.header("⚙️ Configuration")
    checkpoint_dir = Path("./checkpoints")
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        st.error("❌ No checkpoints found in `./checkpoints/` folder. Train a model first!")
        st.info("Run: `python .\run_train.py --data-dir .\data\processed --epochs 10 --model-name custom_cnn`")
        return

    checkpoint_names = {str(ckpt): ckpt.name for ckpt in checkpoints}
    selected_ckpt = st.sidebar.selectbox("Select Checkpoint", options=list(checkpoint_names.keys()), format_func=lambda x: checkpoint_names[x])
    topk = st.sidebar.slider("Top-K Predictions", min_value=1, max_value=10, value=5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**: This app uses a CNN trained on the PlantVillage dataset to classify plant diseases.")
    st.sidebar.markdown("**Dataset**: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) | ~50K images, 15 classes")

    with st.spinner("Loading model..."):
        model, classes, model_name, img_size, device = load_checkpoint(selected_ckpt)

    if model is None:
        return

    st.sidebar.success(f"✅ Model loaded: **{model_name}** (input size: {img_size}×{img_size})")
    st.sidebar.info(f"Classes: {len(classes)}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🎨 Sample Images", "📊 Class Info"])

    with tab1:
        st.subheader("Upload a Leaf Image")
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"], help="Upload a clear image of a plant leaf for disease detection")

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Input Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True, caption=uploaded_file.name)
                st.text(f"Size: {image.size[0]}×{image.size[1]} px")

            with col2:
                st.subheader(f"Top {topk} Predictions")
                with st.spinner("Analyzing..."):
                    results = predict_image(model, image, classes, img_size, device, topk)

                for rank, (disease_class, prob) in enumerate(results, 1):
                    col_rank, col_bar, col_prob = st.columns([0.5, 2, 0.8])
                    with col_rank:
                        st.write(f"**#{rank}**")
                    with col_bar:
                        st.progress(prob)
                    with col_prob:
                        st.write(f"{disease_class}")
                        st.caption(f"{prob*100:.1f}%")

                top_disease, top_prob = results[0]
                st.markdown("---")
                st.success(f"🎯 **Predicted Disease**: {top_disease} ({top_prob*100:.1f}%)")

    with tab2:
        st.subheader("Sample Images from Dataset")
        data_dir = Path("./data/processed/val")

        if not data_dir.exists():
            st.warning("No validation data found. Prepare the dataset first.")
            return

        classes_available = [d.name for d in data_dir.iterdir() if d.is_dir()]
        selected_class = st.selectbox("Select a class to view samples", classes_available)

        class_dir = data_dir / selected_class
        images_available = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

        if not images_available:
            st.info("No images found for this class.")
            return

        num_samples = min(len(images_available), 6)
        cols = st.columns(3)

        for idx in range(num_samples):
            img_path = images_available[idx]
            with cols[idx % 3]:
                sample_img = Image.open(img_path).convert('RGB')
                st.image(sample_img, use_column_width=True, caption=img_path.name)
                with st.spinner("..."):
                    sample_results = predict_image(model, sample_img, classes, img_size, device, topk=1)
                    pred_class, pred_prob = sample_results[0]
                    st.caption(f"🤔 Predicted: {pred_class}\n({pred_prob*100:.0f}%)")

    with tab3:
        st.subheader("Disease Classes in Dataset")
        st.info(f"Total classes: **{len(classes)}**")
        cols = st.columns(3)
        for idx, disease in enumerate(sorted(classes)):
            with cols[idx % 3]:
                st.write(f"• {disease}")


if __name__ == "__main__":
    main()
