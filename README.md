# **Potato Disease Detection**
This repository contains a machine learning project for detecting potato diseases using image data. The dataset includes images of potato plants with early blight, late blight, and healthy conditions. The project uses a Convolutional Neural Network (CNN) to classify the images into these categories.

# **Project Overview**
The goal of this project is to build a deep learning model that can identify diseases in potato plants based on images. Early detection of diseases like early blight and late blight can help prevent significant crop damage.

# **Methodology**
## **Data Collection:**
The dataset used for this project is sourced from the PlantVillage dataset, consisting of 2,152 images, split into three categories:

* Potato Early Blight
* Potato Late Blight
* Healthy
  
## **Data Preprocessing:**
The images were resized to 256x256 pixels.
The dataset was loaded using TensorFlow's image_dataset_from_directory function with batch processing and shuffling.

## **Model Architecture:**
A Convolutional Neural Network (CNN) was built using TensorFlow and Keras.
The model consists of multiple convolutional layers followed by max pooling and fully connected layers.
The model is trained for 50 epochs using a batch size of 32.

## **Training:**
The model was trained on the dataset with categorical cross-entropy as the loss function.
Adam optimizer was used to optimize the learning process.
Model performance was evaluated based on accuracy, and the confusion matrix was used to assess classification errors.

## **Evaluation:**
The model's performance was evaluated on validation data, with the primary metric being accuracy.

# **Libraries Used**
* TensorFlow
* Keras
* numpy
* matplotlib

# **Acknowledgments**
Special thanks to Dhaval Patel and Codebasics for the inspiration and guidance for this project. The dataset was sourced from the PlantVillage dataset on Kaggle.
