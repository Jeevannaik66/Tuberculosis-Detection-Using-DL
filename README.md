# Tuberculosis Detection Using Deep Learning

Welcome to the Tuberculosis Detection project repository! This project leverages deep learning techniques to automate the detection of tuberculosis (TB) from chest X-ray images, offering real-time, accurate results and visual interpretations for medical professionals.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to develop a user-friendly web-based system that uses advanced deep learning (CNN) models to detect tuberculosis (TB) from chest X-ray images. The system enables real-time feedback, allowing users to instantly receive diagnostic results, improving early detection and timely treatment. By automating TB detection, the solution targets regions with limited healthcare resources, providing quick and reliable diagnoses to aid healthcare professionals.

## Objectives

- **User-Friendly Interface**: Design a web interface for easy uploading and analysis of chest X-ray images.
- **Deep Learning Model**: Implement a convolutional neural network (CNN) to detect TB from chest X-rays.
- **Real-Time Diagnosis**: Display the TB diagnosis along with a confidence score.
- **Visualization**: Highlight TB-affected areas in X-ray images using Grad-CAM and SHAP/LIME for interpretable results.

## Features

- **Real-Time TB Detection**: Automatically detects TB from uploaded chest X-ray images.
- **Confidence Score**: Provides a confidence score indicating the likelihood of TB detection.
- **Interpretable Results**: Uses Grad-CAM and SHAP/LIME to highlight affected lung regions for better understanding.
- **Scalable and Accessible**: Designed for deployment in underserved regions to support early detection.
- **Web Interface**: Simple and intuitive interface for healthcare professionals to upload X-ray images and receive instant feedback.

## Usage

- Upload a chest X-ray image through the web interface.
- Get a real-time diagnosis indicating whether TB is present, along with a confidence score.
- View the affected regions in the X-ray using Grad-CAM and SHAP/LIME visualization techniques.
- Use the system for both clinical and research purposes to detect TB early.

## Installation

### Follow these steps to set up and run the Tuberculosis Detection System on your local machine.

### Step 1: Clone the Repository
### ---------------------------------
### Clone the GitHub repository and navigate into the project directory.
- git clone https://github.com/Jeevannaik66/Tuberculosis-Detection-Using-DL.git
- cd Tuberculosis-Detection-Using-DL

### Step 2: Set up a Virtual Environment (Optional but Recommended)
### ----------------------------------------------------------------
### For Linux/Mac:
- python3 -m venv venv
- source venv/bin/activate

### For Windows:
- python -m venv venv
- venv\Scripts\activate

### Step 3: Install the Required Dependencies
### ------------------------------------------
- pip install -r requirements.txt

### Step 4: Download or Link the Model Weights
### -------------------------------------------
### Place the pre-trained model weights in the models/ directory.
### Alternatively, follow the instructions provided in the repository to download the model weights.

### Step 5: Run the Application
### ----------------------------
- python app.py

### Step 6: Access the Application
### -------------------------------
### Open your web browser and navigate to:
### http://127.0.0.1:5000
### You can see the home page and then navigate to the upload page using the navigation bar. 
### Upload a chest X-ray image to use the TB detection system.


## Screenshots

### Home Page
Here’s a screenshot of the **Home page**. You can navigate to the upload page through the navigation bar.

![Home Page](https://github.com/user-attachments/assets/89937b69-6fe3-402a-a131-c93d70c7e50b)




### Upload Page
Once on the upload page, you can upload chest X-ray images for TB detection.

![Upload Page](https://github.com/user-attachments/assets/6119a142-10b7-4703-841d-68d1e3bea88f)




### About Page
The **About page** provides information about the objectives.

![About Page](https://github.com/user-attachments/assets/c7ccfe4d-8db7-412d-8af6-100f282d2d80)




### Prevention Page
The **Prevention page** gives details about tuberculosis prevention methods.

![Prevention Page](https://github.com/user-attachments/assets/856f6f93-b065-47ad-955b-b873ef929ef4)




## Technologies Used

- **Deep Learning Framework**: TensorFlow, Keras
- **Web Development**: Flask, HTML, CSS, JavaScript
- **Visualization Tools**: Grad-CAM, SHAP, LIME
- **Version Control**: Git, GitHub

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, feel free to open an issue or submit a pull request. Your contributions will help improve the accuracy and usability of the system.

## License

This project is licensed under the MIT License.  
Feel free to clone or fork the repository and customize it as per your needs. Thank you for visiting!
