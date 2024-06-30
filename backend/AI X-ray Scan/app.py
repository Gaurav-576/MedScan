import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

CKPT_PATH = 'model/model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

class DenseNet121(nn.Module):
    def _init_(self, out_size):
        super(DenseNet121, self)._init_()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def load_model(ckpt_path):
    model = DenseNet121(N_CLASSES)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

def predict(image, model):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    crops = transform(image)
    crops = crops.view(-1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(crops)
        output_mean = output.view(-1, 10, N_CLASSES).mean(1)
    
    return output_mean.numpy().flatten()

# Streamlit App
st.title("X-Ray Image Upload and Disease Prediction")

st.header("Upload an X-Ray Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        model = load_model(CKPT_PATH)

        if st.button('Predict'):
            predictions = predict(image, model)
            st.header("Predictions")
            for i, pred in enumerate(predictions):
                st.write(f"{CLASS_NAMES[i]}: {pred:.4f}")

    except FileNotFoundError:
        st.error(f"Checkpoint file not found at {CKPT_PATH}. Please check the file path.")
    except Exception as e:
        st.error(f"Error: {e}")