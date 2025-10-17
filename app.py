import streamlit as st
import os
import random
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Auto Image Loader", layout="wide")
st.title("Auto Image Loader & Viewer")

# -----------------------------
# User Inputs
# -----------------------------
webpage_url = st.text_input("Enter webpage URL containing images:")
num_images_to_download = st.number_input("Number of images to download:", min_value=1, value=50)
test_ratio = st.slider("Test set ratio:", 0.0, 0.5, 0.2)
image_size = (224, 224)

output_folder = "images_dataset"
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# -----------------------------
# Scrape Image URLs
# -----------------------------
def scrape_image_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    urls = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if src and src.lower().endswith((".jpg", ".jpeg", ".png")):
            if not src.startswith("http"):
                src = url.rstrip("/") + "/" + src.lstrip("/")
            urls.append(src)
    return urls

# -----------------------------
# Download & Resize Images
# -----------------------------
def download_and_resize(urls, folder):
    images_list = []
    filenames = []
    for i, url in enumerate(tqdm(urls, desc=f"Downloading to {folder}")):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize(image_size)
            img_path = os.path.join(folder, f"image_{i+1}.jpg")
            img.save(img_path)
            images_list.append(img)
            filenames.append(f"image_{i+1}.jpg")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return images_list, filenames

# -----------------------------
# Prepare for ResNet50
# -----------------------------
@st.cache_resource
def prepare_resnet_features(images_list):
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    features = []
    for img in images_list:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x, verbose=0)
        features.append(feat.flatten())
    return np.array(features)

# -----------------------------
# Main Execution
# -----------------------------
if st.button("Load Images"):
    st.info("Scraping image URLs...")
    all_urls = scrape_image_urls(webpage_url)
    st.write(f"Found {len(all_urls)} images on the webpage.")

    if len(all_urls) < num_images_to_download:
        st.warning("Not enough images found. Using all available images.")
        selected_urls = all_urls
    else:
        selected_urls = random.sample(all_urls, num_images_to_download)

    train_urls, test_urls = train_test_split(selected_urls, test_size=test_ratio, random_state=42)

    st.info("Downloading and resizing train images...")
    train_images, train_filenames = download_and_resize(train_urls, train_folder)

    st.info("Downloading and resizing test images...")
    test_images, test_filenames = download_and_resize(test_urls, test_folder)

    st.success(f"Downloaded {len(train_images)} train images and {len(test_images)} test images.")

    # Display Images
    st.subheader("Train Images")
    cols = st.columns(5)
    for idx, img in enumerate(train_images):
        with cols[idx % 5]:
            st.image(img, use_column_width=True, caption=train_filenames[idx])

    st.subheader("Test Images")
    cols = st.columns(5)
    for idx, img in enumerate(test_images):
        with cols[idx % 5]:
            st.image(img, use_column_width=True, caption=test_filenames[idx])

    # Extract Features
    if st.button("Extract ResNet50 Features"):
        st.info("Extracting features, please wait...")
        train_features = prepare_resnet_features(train_images)
        test_features = prepare_resnet_features(test_images)
        st.success("Feature extraction completed!")
        st.write(f"Train features shape: {train_features.shape}")
        st.write(f"Test features shape: {test_features.shape}")
