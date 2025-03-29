import streamlit as st
import cv2
import numpy as np
from retinaface import RetinaFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import base64
from cryptography.fernet import Fernet
import torch
from torch import nn
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

class SecureFaceRecognition:
    def __init__(self, database_path="secure_face_database", encryption_key_path="secure_face_database/encryption_key.key"):
        self.recognition_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.retinaface_model_path = "models/retinaface_resnet50.pth"
        self.database_path = database_path
        self.encryption_key_path = encryption_key_path
        os.makedirs(database_path, exist_ok=True)
        self.key = self.load_or_generate_key()
        self.cipher = Fernet(self.key)
        self.face_database = self.load_database()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_or_generate_key(self):
        if os.path.exists(self.encryption_key_path):
            with open(self.encryption_key_path, 'rb') as key_file:
                return key_file.read()
        else:
            key = Fernet.generate_key()
            with open(self.encryption_key_path, 'wb') as key_file:
                key_file.write(key)
            return key

    def load_database(self):
        db_path = f"{self.database_path}/faces.pkl"
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                encrypted_data = f.read()
                try:
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    return pickle.loads(decrypted_data)
                except Exception as e:
                    print("Decryption failed:", str(e))
                    return {}
        return {}

    def save_database(self):
        data = pickle.dumps(self.face_database)
        encrypted_data = self.cipher.encrypt(data)
        with open(f"{self.database_path}/faces.pkl", 'wb') as f:
            f.write(encrypted_data)

    def detect_faces(self, image):
        faces = RetinaFace.detect_faces(image)
        results = []
        for key in faces:
            face = faces[key]
            facial_area = face['facial_area']
            x1, y1, x2, y2 = facial_area
            face_image = image[y1:y2, x1:x2]
            results.append((face_image, facial_area))
        return results

    def extract_embeddings(self, face_image):
        face_image = self.transform(face_image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.recognition_model(face_image)
        return embedding.numpy()

    def register_face(self, name, encoded_image):
        image_data = base64.b64decode(encoded_image)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        faces = self.detect_faces(image)
        if not faces:
            return "No face detected. Please try again."
        face_image, _ = faces[0]
        embedding = self.extract_embeddings(face_image)
        self.face_database[name] = embedding
        self.save_database()
        return "Face registered successfully."

    def verify_face(self, encoded_image):
        image_data = base64.b64decode(encoded_image)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        faces = self.detect_faces(image)
        if not faces:
            return "No face detected. Please try again."
        face_image, _ = faces[0]
        embedding = self.extract_embeddings(face_image)

        best_match = None
        best_score = -1

        for name, db_embedding in self.face_database.items():
            score = cosine_similarity(embedding, db_embedding)[0][0]
            if score > best_score:
                best_score = score
                best_match = name

        if best_score > 0.7:
            return f"Access granted. Welcome, {best_match}!"
        else:
            return "Access denied. Face not recognized."

# Initialize pipeline
pipeline = SecureFaceRecognition()

# Streamlit UI
st.title("Secure Face Recognition with PyTorch")
choice = st.radio("Choose an option:", ("Register Face", "Login with Face"))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    if choice == "Register Face":
        name = st.text_input("Enter your name:")
        if st.button("Register") and name:
            result = pipeline.register_face(name, encoded_image)
            st.success(result)
    elif choice == "Login with Face":
        if st.button("Login"):
            result = pipeline.verify_face(encoded_image)
            st.success(result)
