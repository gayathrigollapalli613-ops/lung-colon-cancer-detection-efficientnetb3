# Lung and Colon Cancer Detection Using EfficientNetB3

## Project Overview

This project presents a deep learning-based system for classifying histopathological images related to lung and colon cancer using the EfficientNetB3 architecture.

The application is developed using Python and Flask and provides a web interface where an image can be uploaded for prediction.

## Objectives

* Classify histopathological images into cancer-related categories.
* Preprocess images before model prediction.
* Use EfficientNetB3 for image classification.
* Provide predictions through a Flask web application.
* Display the predicted class and confidence percentage.

## Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB3
* NumPy
* OpenCV
* Flask
* HTML
* CSS
* JavaScript

## Project Structure

```text
lung-colon-cancer-detection-efficientnetb3/
│
├── README.md
├── src/
│   ├── app.py
│   └── index.html
│
└── docs/
    ├── Project_Report.docx
    ├── Project_Report.pdf
    ├── Project_Presentation.pptx
    ├── Readme.pdf
    └── Software_Requirements.pdf
```

## Application

The Flask application allows the user to upload an image and obtain a model prediction along with a confidence percentage.

## How It Works

1. User uploads an image.
2. The image is resized to the required input dimensions.
3. The image is converted into an array.
4. Pixel values are normalized.
5. The EfficientNetB3 model processes the image.
6. The predicted class and confidence score are displayed.

## Project Documentation

Detailed project documentation and presentation are available in the `docs` folder.

## Disclaimer

This project is an academic machine learning project and is not intended to provide medical diagnosis or replace professional medical advice.
