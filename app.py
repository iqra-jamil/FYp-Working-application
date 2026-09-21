import streamlit as st #web interface
from ultralytics import YOLO 
from PIL import Image, ImageFile 
import numpy as np
import time
import io
import os
import csv
import json
import shutil
#import subprocess
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


ImageFile.LOAD_TRUNCATED_IMAGES = True
# Ai based smart waste mangement and classification system
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
* { font-family: 'Nunito', sans-serif; }
.header {
    background-color: #2e7d32;
    padding: 18px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 30px;
}
.header h1 {
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin: 0;
}
.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #fff;
    margin-top: 30px;
    margin-bottom: 10px;
}


[data-testid="InputInstructions"] { 
            visibility: hidden !important; 
            }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <span style="font-size:36px;">♻️</span>
    <h1>AI-Based Smart Waste Management and Classification System</h1>
</div>
""", unsafe_allow_html=True)

# constant
MODEL_PATH      = "yolo_trash_classifier.pt"
LOG_FILE        = "classification_log.csv"
ADMIN_PASSWORD  = st.secrets["ADMIN_PASSWORD"]
CLASS_NAMES     = ['glass', 'metal', 'organic', 'paper', 'plastic']

disposal_guide = {
    "plastic": "Please place in the recycling bin.",
    "glass":   "Please place in the glass recycling bin.",
    "metal":   "Please place in the metal recycling bin.",
    "organic": "Please compost this item.",
    "paper":   "Please place in the paper recycling bin.",
    "unknown": "Could not classify. Please dispose of responsibly."
}

category_colors = {
    "plastic": {"bg": "#e3f2fd", "border": "#1565c0", "text": "#1565c0", "icon": "🔵"},
    "glass":   {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#2e7d32", "icon": "🟢"},
    "metal":   {"bg": "#fce4ec", "border": "#b71c1c", "text": "#b71c1c", "icon": "🔴"},
    "organic": {"bg": "#f9fbe7", "border": "#827717", "text": "#827717", "icon": "🟡"},
    "paper":   {"bg": "#fff3e0", "border": "#e65100", "text": "#e65100", "icon": "🟠"},
    "unknown": {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333",    "icon": "⚪"},
}

#  helpers 
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)
# in classification log file
def log_result(label, confidence):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "confidence"])
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, f"{confidence:.1f}"])

def read_logs():
    if not os.path.isfile(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

#  session state 
if "history"       not in st.session_state: st.session_state.history       = []
if "admin_logged"  not in st.session_state: st.session_state.admin_logged  = False
if "train_output"  not in st.session_state: st.session_state.train_output  = "" #tarining result
if "eval_done"     not in st.session_state: st.session_state.eval_done     = False
if "eval_metrics"  not in st.session_state: st.session_state.eval_metrics  = None
if "eval_cm"       not in st.session_state: st.session_state.eval_cm       = None


# TABS

tab1, tab2 = st.tabs(["🗑️ Waste Classifier", "🔐 Admin Panel"])

# TAB 1 — USER

with tab1:

    with st.expander("📋 Image Upload Instructions  Click to Expand", expanded=False):
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:white; margin-bottom:12px;">
            For best results, please upload images that match the following guidelines for each waste type:
        </div>
        """, unsafe_allow_html=True)

        instructions = {
            "🔵 Plastic": {
                "color": "#e3f2fd", "border": "#1565c0", "text": "#1565c0",
                "tips": [
                    "Plastic bottles lying on ground or held in hand",
                    "Crushed, dirty, or used plastic bottles",
                    "Real-world backgrounds (floor, outdoor, etc.)",
                    "Avoid pure white studio-style product shots",
                    "Horizontal or angled orientations work best",
                ]
            },
            "🟢 Glass": {
                "color": "#e8f5e9", "border": "#2e7d32", "text": "#2e7d32",
                "tips": [
                    "Glass bottles, jars, or broken glass pieces",
                    "Clear, green, or brown glass items",
                    "Items placed on a surface or held in hand",
                    "Avoid reflective backgrounds that hide the glass shape",
                ]
            },
            "🔴 Metal": {
                "color": "#fce4ec", "border": "#b71c1c", "text": "#b71c1c",
                "tips": [
                    "Aluminum cans, tin cans, or metal scraps",
                    "Crushed or uncrushed cans both work",
                    "Rusty or shiny metal items are fine",
                    "Avoid images with multiple mixed items",
                ]
            },
            "🟡 Organic": {
                "color": "#f9fbe7", "border": "#827717", "text": "#827717",
                "tips": [
                    "Food scraps, fruit peels, vegetable waste",
                    "Leaves, garden waste, or leftover food",
                    "Natural lighting preferred",
                    "Single item or small pile works best",
                ]
            },
            "🟠 Paper": {
                "color": "#fff3e0", "border": "#e65100", "text": "#e65100",
                "tips": [
                    "Newspapers, cardboard boxes, paper bags",
                    "Crumpled or flat paper both acceptable",
                    "Avoid images where paper is mixed with other waste",
                    "Good lighting to distinguish paper texture",
                ]
            },
        }

        for category, info in instructions.items():
            st.markdown(f"""
            <div style="background-color:{info['color']}; border: 2px solid {info['border']};
                 border-radius:10px; padding:14px 18px; margin-bottom:12px;">
                <div style="font-size:16px; font-weight:800; color:{info['text']}; margin-bottom:8px;">{category}</div>
                <ul style="margin:0; padding-left:18px; color:#444; font-size:14px; font-weight:600;">
                    {''.join(f'<li>{tip}</li>' for tip in info['tips'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#f5f5f5; border:2px solid #9e9e9e; border-radius:10px;
             padding:12px 18px; font-size:14px; font-weight:600; color:#555;">
            ⚠️ <strong>General Tips:</strong> Use clear, well-lit images. Make sure the waste item is
            the main focus of the photo. Avoid very dark, blurry, or heavily filtered images for best accuracy.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; font-size:20px; font-weight:800; margin-bottom:20px;'>Upload or Capture an Image of Waste Item</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        upload_btn = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    with col2:
        webcam_btn = st.camera_input("📷 Capture from Webcam")

    image = None
    if upload_btn:
        try:
            image = Image.open(io.BytesIO(upload_btn.getvalue())).convert("RGB")
        except Exception:
            st.error("Could not open image. Please upload a valid JPG or PNG file.")
    elif webcam_btn:
        try:
            image = Image.open(io.BytesIO(webcam_btn.getvalue())).convert("RGB")
        except Exception:
            st.error("Could not process webcam capture. Please try again.")

    if image:
        st.image(image, use_container_width=True)

        with st.spinner("Classifying..."):
            time.sleep(0.5)
            model = load_model()
            results = model.predict(np.array(image))
            top1       = results[0].probs.top1 #predicted class
            confidence = results[0].probs.top1conf.item() * 100

            if confidence < 35:
                predicted = "unknown"
            else:
                predicted = results[0].names[top1].lower()

        color = category_colors.get(predicted, category_colors["unknown"])

        st.markdown(f"""
        <div style="background-color:{color['bg']}; border: 2px solid {color['border']}; border-radius:12px; padding:20px; text-align:center; margin-top:20px;">
            <div style="font-size:14px; font-weight:700; color:#444; margin-bottom:6px;">Classification Result:</div>
            <div style="font-size:36px;">{color['icon']}</div>
            <div style="font-size:32px; font-weight:800; color:{color['text']}; margin:6px 0;">{predicted.capitalize()} Waste</div>
            <div style="font-size:16px; color:#555;">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        guide = disposal_guide.get(predicted, "Please dispose of responsibly.")
        st.markdown(f"""
        <div style="background-color:{color['bg']}; border: 2px solid {color['border']}; border-radius:10px; padding:14px 20px; margin-top:16px; font-weight:700; font-size:15px; color:{color['text']};">
            {color['icon']} <span>Disposal Guidance:</span> {guide}
        </div>
        """, unsafe_allow_html=True)

        log_result(predicted.capitalize(), confidence) #storing into csv file
         #previous classified hstory
        st.session_state.history.append({
            "image": image.copy(),
            "label": predicted.capitalize(),
            "color": color
        })

    if st.session_state.history:
        st.markdown('<div class="section-title">Previous Classified Items:</div>', unsafe_allow_html=True)
        recent = st.session_state.history[-4:]
        cols = st.columns(min(len(recent), 4))
        for i, item in enumerate(recent):
            with cols[i % 4]:
                st.image(item["image"], use_container_width=True)
                c = item.get("color", category_colors["unknown"])
                st.markdown(f'<div style="text-align:center; font-size:13px; font-weight:700; color:{c["text"]}; background:{c["bg"]}; border:1px solid {c["border"]}; border-radius:6px; padding:3px 0; margin-top:4px;">{c["icon"]} {item["label"]}</div>', unsafe_allow_html=True)


# TAB 2 — ADMIN
with tab2:

    #  login
    if not st.session_state.admin_logged:
        st.markdown("<div style='font-size:20px; font-weight:800; margin-bottom:16px;'>🔐 Admin Login</div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_logged = True
                st.rerun()
            else:
                st.error("Wrong password.")
    else:
        st.markdown("<div style='font-size:20px; font-weight:800; margin-bottom:20px;'>⚙️ Admin Panel</div>", unsafe_allow_html=True)

        if st.button("🚪 Logout"):
            st.session_state.admin_logged = False
            st.rerun()

        st.markdown("---")

       
        # 4.1 + 4.2  — Dataset upload & retrain
      
        st.markdown("### 📁 Dataset Management & Retraining")
        st.markdown("""
        <div style="background:#e8f5e9; border:2px solid #2e7d32; border-radius:10px; padding:14px 18px; font-size:14px; font-weight:600; color:#2e7d32; margin-bottom:16px;">
            Upload your dataset as a <strong>ZIP file</strong>. It must contain three folders:
            <code>train/</code>, <code>val/</code>, and <code>test/</code>.<br>
            Each split folder should have sub-folders named after the classes:
            <code>glass</code>, <code>metal</code>, <code>organic</code>, <code>paper</code>, <code>plastic</code>.
        </div>
        """, unsafe_allow_html=True)

        uploaded_zip = st.file_uploader("📦 Upload Dataset ZIP", type=["zip"])

        epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=10)

        if st.button("🚀 Start Retraining"):
            data_dir = None
            #uploaded zip file handling
            if uploaded_zip is not None:
                import zipfile
                extract_path = "uploaded_dataset"
                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path) #dlt
                os.makedirs(extract_path, exist_ok=True)
                zip_bytes = io.BytesIO(uploaded_zip.read())
                with zipfile.ZipFile(zip_bytes, "r") as z:
                    z.extractall(extract_path)
                entries = os.listdir(extract_path)
                if len(entries) == 1 and os.path.isdir(os.path.join(extract_path, entries[0])):
                    data_dir = os.path.join(extract_path, entries[0])
                else:
                    data_dir = extract_path
             #checking dataset validation
            if not data_dir or not os.path.isdir(data_dir):
                st.error("Please upload a ZIP file to start training.")
            else:
                splits = [s for s in ["train", "val", "test"] if os.path.isdir(os.path.join(data_dir, s))]
                if len(splits) < 2:
                    st.error("Dataset folder must contain at least train/ and val/ sub-folders.")
                else:
                    st.info(f"Found splits: {splits}  |  Training for {epochs} epochs...")

                    try:
                        #import torch
                        train_model = YOLO("yolo11n-cls.pt")

                        with st.spinner("Training in progress — this may take a while..."):
                            train_results = train_model.train(
                                data=data_dir,
                                epochs=epochs,
                                imgsz=224,
                                batch=16,
                                device=0 if __import__("torch").cuda.is_available() else "cpu",
                                workers=0,
                                verbose=False,
                            )

                        train_model.save(MODEL_PATH)
                        load_model.clear()

                        top1_val = train_results.results_dict.get("metrics/accuracy_top1", None)
                        msg = f"✅ Training complete! Best val top-1 accuracy: {top1_val:.4f}" if top1_val else "✅ Training complete! Model saved."
                        st.success(msg)
                        st.session_state.train_output = msg

                    except Exception as e:
                        st.error(f"Training failed: {e}")
    
        if st.session_state.train_output:
            st.markdown(f"""
            <div style="background:#f1f8e9; border:2px solid #558b2f; border-radius:8px; padding:12px 16px; font-size:14px; font-weight:700; color:#33691e; margin-top:8px;">
                {st.session_state.train_output}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

       
        # 4.3  Classification log
   
        st.markdown("### 📊 Classification Log")

        logs = read_logs()
        if not logs:
            st.info("No classifications have been logged yet.")
        else:
            import pandas as pd
            df = pd.DataFrame(logs)
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

            st.markdown(f"**Total classifications logged: {len(df)}**")
            st.dataframe(df.sort_values("timestamp", ascending=False).reset_index(drop=True), use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Log as CSV", data=csv_bytes, file_name="classification_log.csv", mime="text/csv")

        st.markdown("---")

       
        # 4.4  Model evaluation report
     
        st.markdown("### 📈 Model Evaluation Report")

        if st.button("🔍 Run Evaluation"):
            if not os.path.isfile(MODEL_PATH):
                st.error(f"Model file '{MODEL_PATH}' not found. Make sure it is in the same folder as app.py.")
            else:
                data_path = None
                if os.path.isdir("uploaded_dataset"):
                    candidates = ["uploaded_dataset"] + [
                        os.path.join("uploaded_dataset", d)
                        for d in os.listdir("uploaded_dataset")
                        if os.path.isdir(os.path.join("uploaded_dataset", d))
                    ]
                    for c in candidates:
                        if all(os.path.isdir(os.path.join(c, s)) for s in ["train", "val", "test"]):
                            data_path = c
                            break
# performing evaluation
                if not data_path:
                    st.error("Dataset not found. Please upload your dataset ZIP in the Dataset Management section above first, then click Run Evaluation.")
                else:
                    with st.spinner("Running evaluation on test split..."):
                        try:
                            eval_model = YOLO(MODEL_PATH)

                            # evaluate 
                            m_test = eval_model.val(
                                data=data_path,
                                split="test",
                                verbose=False,
                                deterministic=True,
                                seed=42
                            )

                            raw = m_test.confusion_matrix.matrix
                            combined_cm = raw[:len(CLASS_NAMES), :len(CLASS_NAMES)].astype(int)

                            total_correct = int(np.trace(combined_cm))
                            total_images  = int(combined_cm.sum())
                            overall_acc   = total_correct / total_images if total_images > 0 else 0

                            per_class = {}
                            for i, cls in enumerate(CLASS_NAMES):
                                TP = combined_cm[i, i]
                                FP = combined_cm[:, i].sum() - TP
                                FN = combined_cm[i, :].sum() - TP
                                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                                recall    = TP / (TP + FN) if (TP + FN) != 0 else 0
                                f1        = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
                                per_class[cls] = {
                                    "precision": round(float(precision), 4),
                                    "recall":    round(float(recall),    4),
                                    "f1":        round(float(f1),        4),
                                }

                            st.session_state.eval_metrics = {
                                "top1":      round(float(overall_acc), 4),
                                "per_class": per_class,
                            }
                            st.session_state.eval_cm   = combined_cm
                            st.session_state.eval_done = True

                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")

        if st.session_state.eval_done and st.session_state.eval_metrics:
            m = st.session_state.eval_metrics

            #  overall metrics
            avg_precision = np.mean([v["precision"] for v in m["per_class"].values()]) * 100
            avg_recall    = np.mean([v["recall"]    for v in m["per_class"].values()]) * 100
            avg_f1        = np.mean([v["f1"]        for v in m["per_class"].values()]) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Accuracy",  f"{m['top1']*100:.2f}%")
            col2.metric("⚡ Precision", f"{avg_precision:.2f}%")
            col3.metric("📊 Recall",    f"{avg_recall:.2f}%")
            col4.metric("🎖️ F1-Score",  f"{avg_f1:.2f}%")

            #  confusion matrix 
            st.markdown("**Confusion Matrix (Test Split)**")
            fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
            cm_int = st.session_state.eval_cm.astype(int)
            sns.heatmap(
                cm_int,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted", fontsize=12)
            ax_cm.set_ylabel("Actual",    fontsize=12)
            ax_cm.set_title("Confusion Matrix — Test Split", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

# footer
st.markdown("""
<div style="text-align:center; margin-top:40px; font-size:13px; color:#888; font-weight:600;">
    ♻️ Smart Waste Classifier — Helping build a cleaner world<br>
    <span style="font-size:12px; color:#aaa;">Developed by: Iqra jamil &nbsp;|&nbsp; Student ID: Bc220200221 &nbsp;|&nbsp; Final Year Project 2025</span>
</div>
""", unsafe_allow_html=True)
