# TrafficSignClassifierHW

# Self Driving Car Project 2 - Traffic Sign Classifier
This doc describes the work I did for the Udacity Self Driving Car Traffic Sign Classification Project.

The Jupyter notebook Traffic_Sign_Classifier contains all the code and the results of the 
execution of that code.

The challenge of this project is to train a convolutional neural network model to classify German traffic signs. Like this

[//]: # (Image References)

[image1]: ./origImages/german_1.jpg "Road Work"
[image2]: ./origImages/speed50.jpg "Speed Limit 50kph"

![Road Work][image1]

and this:

![Speed Limit 50kph][image2]

## Build a Traffic Sign Recognition Project

To accomplish this task, I built an image processing pipeline, which consisted of the following steps.

* Read train, test and validation image sets
* Perform some basic analysis of the images
* Explore and summarize the images
* Convert all images to grayscale
* Augment the images by transforming the original images in various ways
* Normalize the images
* Create a CNN model architecture
* Train, test and validate model
* Test model on new & unseen images
* Analyze performance of the model
* Analyze the softmax probabilities of the new iamges
* Summarize the results in a written report

## Rubric Points

### Files Submitted

[iPython Notebook](https://github.com/mjg042/TrafficSignClassifierHW/blob/master/Traffic_Sign_Classifier.ipynb)

[HTML output of notebook](https://github.com/mjg042/TrafficSignClassifierHW/blob/master/Traffic_Sign_Classifier.html)

[Project writeup (this file)](https://github.com/mjg042/TrafficSignClassifierHW/blob/master/README.md)

### Dataset Exploration

Training, testing and validation images sets were provided by Udacity.

#### Dataset Summary

All images were 32x32x3 in size. There were:

* 34,799 training images 
* 12,630 testing images, and
* 4,410 validation images

43 different traffic signs were represented in the dataset. Each type of sign represents a class that the model
will be expected to identify.


#### Exploratory Visualization

[//]: # (Image References)

[image3]: ./origImages/origImages.jpg "Random Images from dataset"

![Random Images from dataset][image3]





### Design & Test Model Architecture

#### Preprocessing
#### Model Architecture
#### Model Training
#### Solution Approach


### Test Model on New Images

#### Acquiring New Images
#### Performance on New Images
#### Model Certainty - Softmax Probabilities



