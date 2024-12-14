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

  

### Installation

Follow these steps to set up and run the **Tuberculosis Detection System** on your local machine.

### Step 1: Clone the Repository
---------------------------------
- Clone the GitHub repository and navigate into the project directory:
  ```bash
  git clone https://github.com/Jeevannaik66/Tuberculosis-Detection-Using-DL.git
  cd Tuberculosis-Detection-Using-DL
  ```

### Step 2: Install Git LFS (Large File Storage)
-----------------------------------------------
Git LFS is used to manage large files (such as model weights) in this repository. To download these large files, you need to install Git LFS.

  - **For Windows**:
    - Download the Git LFS installer from [Git LFS website](https://git-lfs.github.com/).
    - Run the installer and follow the instructions.
    - After installation, run the following command in the terminal or command prompt:
      ```bash
      git lfs install
      ```

  - **For macOS**:
    - If you have Homebrew installed, run the following:
      ```bash
      brew install git-lfs
      git lfs install
      ```

  - **For Linux (Ubuntu/Debian)**:
    - Use the following command:
      ```bash
      sudo apt-get install git-lfs
      git lfs install
      ```

### Step 3: Pull the Git LFS Files
---------------------------------
- After cloning the repository and installing Git LFS, pull the large model files using the following command:
  ```bash
  git lfs pull
  ```
  This will download the large files (e.g., the model weights) into your local machine.

#### Alternative Step: Manual Download of Model Weights
- If the `git lfs pull` command fails due to bandwidth limitations, you can download the `precheck_model.h5` and `tb_classification_model.h5` manually.
- Use these [Google Drive links](https://drive.google.com/drive/folders/1UpqY3iRSj9jFPzi1Gqp5HRXZ9CG9Vgai?usp=drive_link) to download both model files.
- Download the following models:
  - `precheck_model.h5`
  - `tb_classification_model.h5`
- After downloading, place these files into the `model/` directory of the project.

### Step 4: Set up a Virtual Environment (Optional but Recommended)
-------------------------------------------------------------------
- **For Linux/Mac**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **For Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### Step 5: Install the Required Dependencies
--------------------------------------------
- Install all the required dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Step 6: Run the Application
----------------------------
- To run the application, execute the following command:
  ```bash
  python app.py
  ```

### Step 7: Access the Application
-------------------------------
- Open your web browser and navigate to:
  ```
  http://127.0.0.1:5000
  ```
- You can see the home page and then navigate to the upload page using the navigation bar.
- Upload a chest X-ray image to use the TB detection system.

---

### Troubleshooting Tips:
- If you experience any issues with Git LFS or the model files, ensure that both model files (`precheck_model.h5` and `tb_classification_model.h5`) are placed correctly in the `model/` directory. If you continue to face problems with Git LFS, manually downloading the models from Google Drive and placing them in the appropriate folder should resolve the issue.


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
Feel free to clone or fork the repository and customize it as per your needs. Thank you for visiting!
