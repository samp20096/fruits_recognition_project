# Fruits-360 Recognition App

A Streamlit-based web application for classifying fruits and vegetables using a custom deep learning model trained on the Fruits-360 dataset. You can upload an image from your device or use your device's camera to take a picture and identify the fruit from 250 different classes.

## Prerequisites

Make sure you have Python installed (preferably Python 3.8 to 3.11). 

## Setup Instructions

To get this application running locally, please follow these steps carefully:

### 1. Clone the repository

First, clone this repository to your local machine and navigate into the project folder.

```bash
git clone https://github.com/samp20096/fruits_recognition_project
cd fruits360_project
```

### 2. Install Dependencies

It is recommended to use a virtual environment. Install the required libraries using pip:

```bash
pip install streamlit tensorflow numpy Pillow
```

### 3. Download the Model

The app requires the pre-trained model file to function.

1. Download the `fruits_360.keras` model file.
2. Place the `fruits_360.keras` file directly in the same folder as `app.py` (the root of the project). 

If the model is missing, the application will display an error message and refuse to run.

### 4. Run the Application

Once the repository is cloned, dependencies are installed, and the model is placed in the correct directory, you can launch the app:

```bash
streamlit run app.py
```

This will start a local server, and your default web browser will automatically open the app. If it doesn't open automatically, you can navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## How to use

1. Select either to **Take a picture** using your camera or **upload from gallery**.
2. Click **Save image to computer** if you wish to save a local copy of the picture.
3. Click the **Fruit Recognition** button.
4. The model will analyze the 100x100 resized version of the image and predict the fruit class!
