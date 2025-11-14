🌿 Plant Disease Classifier

A Statistical Machine Learning project that detects plant diseases from leaf images using a trained deep learning model.

📌 Overview

Plant diseases significantly reduce agricultural productivity. Early detection helps farmers take preventive measures and reduce crop loss.
This project uses PyTorch, CNN-based feature extraction, and Streamlit to build an end-to-end web app for disease prediction from plant leaf images.

🚀 Features

📸 Upload or capture leaf image

🔍 Deep learning–based disease detection

📊 Prediction probability chart

🧪 Model interpretation using Grad-CAM (saliency map)

🌐 Streamlit-based user interface

💾 Configurable model loading

🧠 Model & Methodology
1. Dataset

The model is trained on the PlantVillage dataset from Kaggle (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

Contains 38 classes of healthy and diseased leaf images.

2. Preprocessing

Image normalization

Resizing to 224×224

Data augmentation for robustness

Standard PyTorch transforms

3. Model Architecture

Backbone: ResNet 152

Fully connected classifier head fine-tuned for plant leaf disease categories

Loss: Cross-entropy

Optimizer: Adam 

Evaluation metrics: Accuracy, Loss, Confusion Matrix

4. Training

Trained on GPU

Validation split to avoid overfitting

Early stopping and checkpoint saving

5. Deployment

Streamlit application

Upload image → Preprocess → Inference → Prediction results

Optional Grad-CAM visualization

🧪 Results

High classification accuracy on validation data

Clear visualization of model confidence

Grad-CAM highlights infected regions of leaves



🛠️ Tech Stack

Python

PyTorch

Streamlit

Torchvision

Matplotlib / PIL

📦 How to Run Locally
1. Clone the Repository
git clone https://github.com/<your-username>/plant-disease-classifier.git
cd plant-disease-classifier

2. Install Requirements
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run app_streamlit.py

🖥️ Project Structure
├── app_streamlit.py        # Web UI
├── model.pth               # Trained model checkpoint
├── utils/                  # Helper scripts
├── notebooks/              # Training notebook
├── requirements.txt
└── README.md

🌱 Screenshots


📚 Future Improvements

Multi-disease detection per leaf

Larger dataset integration

Model quantization for mobile apps

REST API using FastAPI / Flask

🤝 Contributors

Jeevan Prakash Meghwal (Project Lead)

Gauranvi Mehra 

Taanya Raawat


⭐ Support

If you found this project helpful, please star the repository ⭐
It motivates further development!
