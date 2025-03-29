# Secure Face Recognition with PyTorch

This project implements a secure face recognition system using PyTorch, RetinaFace for face detection, and InceptionResnetV1 from `facenet-pytorch` for face recognition. It uses AES encryption with `cryptography.fernet` to securely store face embeddings.

## Features
- **Face Registration:** Users can register their faces by uploading an image.
- **Face Verification:** Users can log in using face recognition.
- **Encryption:** All face embeddings are encrypted using AES encryption with a securely stored key.
- **Local Models:** Both face detection and recognition use locally stored models.
- **Streamlit UI:** Simple and interactive user interface built using Streamlit.

---

## Prerequisites

- Python 3.8 or above
- CUDA (for GPU acceleration, optional)
- [Anaconda (Optional)](https://www.anaconda.com/products/distribution)

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/secure-face-recognition.git
    cd secure-face-recognition
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    conda create -n face_recognition python=3.10
    conda activate face_recognition
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the models:
    - **InceptionResnetV1** will be downloaded automatically using `facenet-pytorch`.
    - Download the RetinaFace model (`retinaface_resnet50.pth`) and place it in the `models` directory:
      ```bash
      mkdir models
      # Place retinaface_resnet50.pth here
      ```

---

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

2. Choose an option:
    - **Register Face:** Upload an image and enter your name to register your face.
    - **Login with Face:** Upload an image to verify your identity.

---

## Project Structure
```bash
secure-face-recognition/
├── models/                   # Pretrained models (RetinaFace, InceptionResnetV1)
│   ├── retinaface_resnet50.pth
├── secure_face_database/     # Encrypted database for face embeddings
│   ├── faces.pkl
│   ├── encryption_key.key
├── main.py                   # Main application file
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
