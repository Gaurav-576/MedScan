import os
import cv2
import numpy as np
import streamlit as st
import onnxruntime
import torch
from pathlib import Path
from matplotlib.colors import TABLEAU_COLORS

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# Model and image dimensions
parent_root = Path(__file__).parent.parent.absolute().__str__()
h, w = 640, 640
model_onnx_path = os.path.join(parent_root, "yolov7-p6-bonefracture.onnx")

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predefined colors for visualizing detections
colors = list(TABLEAU_COLORS.values())[:10]

def load_img(uploaded_file):
    """Function to load and decode an uploaded image."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image[..., ::-1]  # Convert BGR to RGB
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def preproc(img):
    """Preprocess image: resize and normalize."""
    try:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32).transpose(2, 0, 1) / 255
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def model_inference(model_path, image_np, device="cpu"):
    """Perform inference using the ONNX model."""
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    try:
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: image_np})
        return output[0][:, :6]  # Return only the relevant output
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

def post_process(img, output, score_threshold=0.3):
    """Post-process the model output and visualize detections."""
    try:
        det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
        id2names = {
            0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
            3: "fracture", 4: "metal", 5: "periostealreaction", 
            6: "pronatorsign", 7: "softtissue", 8: "text"
        }
        H, W = img.shape[:2]
        label_txt = ""
        for idx in range(len(det_bboxes)):
            if det_scores[idx] > score_threshold:
                bbox = det_bboxes[idx]
                label = int(det_labels[idx])
                # Convert bbox format and validate coordinates
                bbox_xywhn = xyxy2xywhn(bbox, H, W)
                label_txt += f"{label} {det_scores[idx]:.5f} " \
                             f"{bbox_xywhn[0]:.5f} {bbox_xywhn[1]:.5f} " \
                             f"{bbox_xywhn[2]:.5f} {bbox_xywhn[3]:.5f}\n"
                bbox_xyxy = xywhn2xyxy(bbox_xywhn, H, W)
                bbox_int = [int(x) for x in bbox_xyxy]
                x1, y1, x2, y2 = bbox_int
                color_map = colors[label]
                txt = f"{id2names[label]} {det_scores[idx]:.2f}"
                # Draw rectangle and text on image
                cv2.rectangle(img, (x1, y1), (x2, y2), color_map, 2)
                (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1 - 2, y1 - text_height - 10), (x1 + text_width + 2, y1), color_map, -1)
                cv2.putText(img, txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img, label_txt
    except Exception as e:
        st.error(f"Error during post-processing: {e}")
        return img, ""

def xyxy2xywhn(bbox, H, W):
    """Convert bounding box format from (x1, y1, x2, y2) to (x_center_norm, y_center_norm, w_norm, h_norm)."""
    x1, y1, x2, y2 = bbox
    return [0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, (x2 - x1) / W, (y2 - y1) / H]

def xywhn2xyxy(bbox, H, W):
    """Convert bounding box format from (x_center_norm, y_center_norm, w_norm, h_norm) to (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]

if __name__ == "__main__":
    st.title("Bone Fracture Detection")
    uploaded_file = st.file_uploader("Choose an image file", type=ALLOWED_EXTENSIONS)
    
    if uploaded_file is not None:
        try:
            conf_thres = st.slider("Object confidence threshold", 0.2, 1.0, step=0.05)
            img = load_img(uploaded_file)
            if img is not None:
                img_pp = preproc(img)
                if img_pp is not None:
                    out = model_inference(model_onnx_path, img_pp, device)
                    if out is not None:
                        out_img, out_txt = post_process(img, out, conf_thres)
                        st.image(out_img, caption="Prediction", channels="RGB")
                        col1, col2 = st.columns(2)
                        col1.download_button(
                            label="Download prediction",
                            data=cv2.imencode(".png", out_img[..., ::-1])[1].tobytes(),
                            file_name=uploaded_file.name,
                            mime="image/png"
                        )
                        col2.download_button(
                            label="Download detections",
                            data=out_txt,
                            file_name=uploaded_file.name[:-4] + ".txt",
                            mime="text/plain"
                        )
        except Exception as e:
            st.error(f"Error: {e}")
