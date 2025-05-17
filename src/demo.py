import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from resNet34 import ResNet34, ResNet, BasicBlock
import os
import sys

# Add ResNet to safe globals
torch.serialization.add_safe_globals([ResNet, BasicBlock])

# Initialize model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet34()
try:

    loaded_model = torch.load(os.path.join("trained_model", "resNET_model.pth"), 
                            map_location=device,
                            weights_only=False) # If the loaded model is a state dict, load it into our model
    if isinstance(loaded_model, dict):
        model.load_state_dict(loaded_model)
    else:
        
        model = loaded_model
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 classes.")


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Debug information
        st.write("Debug Info:")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")

        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File size too large. Please upload an image smaller than 5MB.")
            st.stop()

        # Validate image format
        if uploaded_file.type not in ["image/jpeg", "image/png"]:
            st.error("Invalid file format. Please upload a JPEG or PNG image.")
            st.stop()

        try:
            # Reset file pointer and open image
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            
            # Additional image validation
            if image.format not in ["JPEG", "PNG"]:
                st.error("Invalid image format. Please upload a JPEG or PNG image.")
                st.stop()
                
            st.write(f"Image format: {image.format}")
            st.write(f"Image mode: {image.mode}")
            st.write(f"Image size: {image.size}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess the image
            try:
                image_tensor = transform(image).unsqueeze(0).to(device)
                st.write("Image preprocessing successful")
            except Exception as e:
                st.error(f"Error preprocessing image: {str(e)}")
                st.stop()
            
            # Make prediction
            try:
                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probability = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                st.write(f"Predicted class: {classes[predicted.item()]}")
                
                st.write("Confidence scores:")
                for i, prob in enumerate(probability):
                    st.write(f"{classes[i]}: {prob.item():.2%}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.stop()
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.stop()
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please try uploading a different image.")
        st.write("Full error traceback:")
        st.write(str(sys.exc_info()))