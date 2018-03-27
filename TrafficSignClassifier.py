# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:07:21 2018

used the following as guides
https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb
https://becominghuman.ai/updated-my-99-40-solution-to-udacity-nanodegree-project-p2-traffic-sign-classification-5580ae5bd51f
https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/image/image.py
http://www.scipy-lectures.org/advanced/image_processing/
https://navoshta.com/traffic-signs-classification/

conda install -c https://conda.anaconda.org/jjhelmus tensorflow
 
@author: michaelg@bluemetal.com
"""

"""
wd = "C:\\Users\\mgriffi3\\notebooks\\udacity"
import os
os.chdir(wd)

training_file = 'train.p'
import os.path
os.path.isfile(training_file) 

"""

# Load pickled data
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

# pip install opencv-python
import cv2
import glob
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test,  y_test  =  test['features'],  test['labels']


n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train.shape[1:4]
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


"""
Data Visualization
"""
# Visualizations will be shown in the notebook.
%matplotlib inline


def PlotMultipleImages(nrows, nImages, idx, allImages):
    ncols = int(nImages/nrows)
    #idx = np.random.choice(len(allImages), nImages)
    images = allImages[idx]
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(nrows, np.ceil(nImages/float(nrows)), n+1)
        plt.imshow(image.squeeze())
        a.set_title('Image ' + str(idx[n]))
    fig.set_size_inches(np.array(fig.get_size_inches()) * 2)

#plot_figures(image.squeeze())
    
def PlotMultipleImageHistograms(nrows, nImages, allImages):
    ncols = int(nImages/nrows)
    idx = np.random.choice(len(allImages), nImages)
    images = allImages[idx]
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(nrows, np.ceil(nImages/float(nrows)), n+1)
        plt.hist(image.ravel(), 256, [0,256])
        a.set_title('Image ' + str(idx[n]))
    fig.set_size_inches(np.array(fig.get_size_inches()) * 2)
    
def PlotHistogram(y, n, title):
    hist,bins = np.histogram(y, bins=n)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Label distribution for ' + title + ' data', size=18)
    plt.show()

"""
plot a sample of the images
"""
nrows = 2
nImages = 10
idx = np.random.choice(len(X_train), nImages)
PlotMultipleImages(nrows, nImages, idx, X_train)

"""
plot some intensity distributions 
"""
PlotMultipleImageHistograms(nrows, nImages, X_train)

"""
compare label distribution across y_train, y_test, y_valid
want distributions to be similar across all 3 datasets
"""
PlotHistogram(y_train, n_classes, 'training')
PlotHistogram(y_test, n_classes, 'testing')
PlotHistogram(y_valid, n_classes, 'validation')

"""
Preprocess the data here. It is required to normalize the data. 
Other preprocessing steps could include converting to grayscale, etc.
Feel free to use as many code cells as needed.
"""

# convert to grayscale
# from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    g = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    g = g[:,:,np.newaxis]
    return g
    #return np.sum(rgb/3, axis=3, keepdims=True)

def ConvertToGrayScale(images):
    #shape = (images.shape[0],images.shape[1],images.shape[2])
    grayImages = np.empty(images.shape)
    for n, image in enumerate(images):
        grayImages[n] = rgb2gray(image)
    return grayImages

# not used
def ConvertToGrayScaleAlt(images):
    grayImages = np.sum(images/3, axis=3, keepdims=True)
    return grayImages

"""
# convert 1 image to test

image = X_train[123]
grayimg = rgb2gray(image)
plt.imshow(grayimg)
plt.show()
plt.imshow(image)
plt.show()

"""
#X_train_gray = ConvertToGrayScale(X_train)
#X_test_gray = ConvertToGrayScale(X_test)
#X_valid_gray = ConvertToGrayScale(X_valid)

#X_train_gray = ConvertToGrayScaleAlt(X_train)
#X_test_gray = ConvertToGrayScaleAlt(X_test)
#X_valid_gray = ConvertToGrayScaleAlt(X_valid)

X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)

#X_validation = X_valid_gray
#y_validation = y_valid

print(X_train.shape)
print(X_train_gray.shape)

nrows = 2
nImages = 10
idx = np.random.choice(len(X_train), nImages)
PlotMultipleImages(nrows, nImages, idx, X_train)
PlotMultipleImages(nrows, nImages, idx, X_train_gray)

X_train = X_train_gray
X_test = X_test_gray
X_valid = X_valid_gray


"""
https://www.researchgate.net/post/If_I_used_data_normalization_x-meanx_stdx_for_training_data_would_I_use_train_Mean_and_Standard_Deviation_to_normalize_test_data

###Normalize data by substracting the mean of the entire data set. This serves
###to center the images. 

###Can try normalizing by the stdev too.

I normalized by z-score.

This is done so that each image has a similar range so that gradients computed
for back prop have a similar range. Deep learning methods typically share
many parameters and we want these shared params to be sort of uniform.

Should use train data mean and stdev for test if test data does not vary
much from the training data. If the 2 sets differ significantly, may want
to use mean and stdev of test set to normalize test set.

The distributions of classes (as shown above) indicates that the classes are
reasonably well distributed across the sets.

"""

Xmean = np.mean(X_train)
Xstdv = np.std(X_train)

XnormTrain = (X_train - Xmean)/Xstdv
XnormTest  = (X_test  - Xmean)/Xstdv
XnormValid = (X_valid - Xmean)/Xstdv

print(np.mean(XnormTrain))
print(np.mean(XnormTest))
print(np.mean(XnormValid))

nrows = 1
nImages = 5
idx = np.random.choice(len(X_train), nImages)
PlotMultipleImages(nrows, nImages, idx, X_train)
PlotMultipleImages(nrows, nImages, idx, XnormTrain)

X_train = XnormTrain
X_test = XnormTest
X_valid = XnormValid



"""
image augmentation

mxnet (referenced above) has some great image augmentation code

the general idea is to select a random set of images from the training set
and modify them in semi-random ways, e.g.
   - translate 
   - scale up and then crop back to 32x32
   - warp 
   - modify brightness
   - etc.
   
and then add the modified images back to training set - including y_train
   
"""

def TranslateImage(img):
    # somewhat randomly translate image
    rows, cols, _ = img.shape
    px = 6
    dx, dy = np.random.randint(-px, px, 2)
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    dst = dst[:, :, np.newaxis]
    return dst

def ScaleImage(img):
    # somewhat randomly translate image
    rows, cols, _ = img.shape
    # transform limits
    px = np.random.randint(-6,6)
    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(rows,cols))
    dst = dst[:,:,np.newaxis]
    return dst

def WarpImage(img):
    # somewhat randomly translate image
    rows, cols, _ = img.shape
    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.2   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.2
    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))    
    dst = dst[:,:, np.newaxis]
    return dst

def BrightnessImage(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst

def ContrastImage(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst


def CreateNewSetOfImages(images,method):
    #shape = (images.shape[0],images.shape[1],images.shape[2])
    newImages = np.empty(images.shape)
    if method == 'translate':
        for n, image in enumerate(images):
            newImages[n] = TranslateImage(image)
    elif method == 'scale':
        for n, image in enumerate(images):
            newImages[n] = ScaleImage(image)
    elif method == 'warp':
        for n, image in enumerate(images):
            newImages[n] = WarpImage(image)
    elif method == 'brightness':
        for n, image in enumerate(images):
            newImages[n] = BrightnessImage(image)
    return newImages

# generate new augmented training data
X_train_aug = X_train
y_train_aug = y_train

# randomly select 1/5th of the training set 
nAugment = int(len(X_train)/5)

"""
translate those images and append them to the augmented data sets
"""

idx = np.random.choice(len(X_train_aug), nAugment)
xaug = X_train_aug[idx,:]
yaug = y_train_aug[idx]
xaugTran = CreateNewSetOfImages(xaug, 'translate')

print(xaug.shape)

X_train_aug = np.concatenate((X_train_aug, xaugTran), axis=0)
y_train_aug = np.concatenate((y_train_aug, yaug), axis=0)

nrows = 1
n = 5
idx = np.random.choice(len(xaug), n)
PlotMultipleImages(nrows, n, idx, xaug)
print('translated')
PlotMultipleImages(nrows, n, idx, xaugTran)

"""
scale images
"""
idx = np.random.choice(len(X_train_aug), nAugment)
xaug = X_train_aug[idx,:]
yaug = y_train_aug[idx]
xaugTran = CreateNewSetOfImages(xaug, 'scale')

X_train_aug = np.concatenate((X_train_aug, xaugTran), axis=0)
y_train_aug = np.concatenate((y_train_aug, yaug), axis=0)

nrows = 1
n = 5
idx = np.random.choice(len(xaug), n)
PlotMultipleImages(nrows, n, idx, xaug)
print('scaled')
PlotMultipleImages(nrows, n, idx, xaugTran)

"""
warp images
"""
idx = np.random.choice(len(X_train_aug), nAugment)
xaug = X_train_aug[idx,:]
yaug = y_train_aug[idx]
xaugTran = CreateNewSetOfImages(xaug, 'warp')

X_train_aug = np.concatenate((X_train_aug, xaugTran), axis=0)
y_train_aug = np.concatenate((y_train_aug, yaug), axis=0)

nrows = 1
n = 5
idx = np.random.choice(len(xaug), n)
PlotMultipleImages(nrows, n, idx, xaug)
print('warp')
PlotMultipleImages(nrows, n, idx, xaugTran)


"""
brightness
"""
idx = np.random.choice(len(X_train_aug), nAugment)
xaug = X_train_aug[idx,:]
yaug = y_train_aug[idx]
xaugTran = CreateNewSetOfImages(xaug, 'brightness')

X_train_aug = np.concatenate((X_train_aug, xaugTran), axis=0)
y_train_aug = np.concatenate((y_train_aug, yaug), axis=0)

nrows = 1
n = 5
idx = np.random.choice(len(xaug), n)
PlotMultipleImages(nrows, n, idx, xaug)
print('brighness')
PlotMultipleImages(nrows, n, idx, xaugTran)



print('Now have',len(X_train_aug), 'training image')
print('Now have',len(y_train_aug), 'training labels')

X_train = X_train_aug
y_train = y_train_aug

"""
I chose to keep the original test and validation sets, and augment the
training set. By choosing random sets of training images to transform
I kept approximately the same distribution of labels/classifications
in all three sets.

TODO: revisit this decision if necessary
"""



def LeNetORIG(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    W3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros(120))    
    x = tf.add(tf.matmul(x, W3), b3)
    
    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros(84)) 
    x = tf.add(tf.matmul(x, W4), b4)
    
    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    W5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(43)) 
    logits = tf.add(tf.matmul(x, W5), b5)
    
    return logits


def LeNet2(x):    
    # Hyperparameters
    mu = 0.0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="W2")
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = x
    
    # TODO: Layer 3: Convolutional. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma), name="W3")
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)
    layer3 = x

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2flat = flatten(layer2)
    print("layer2flat shape:",layer2flat.get_shape())
    
    # Flatten x. Input = 1x1x400. Output = 400.
    xflat = flatten(x)
    print("xflat shape:",xflat.get_shape())
    
    # Concat layer2flat and x. Input = 400 + 400. Output = 800
    x = tf.concat([xflat, layer2flat], 1)
    print("x shape:",x.get_shape())
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    
    # TODO: Layer 4: Fully Connected. Input = 800. Output = 43.
    W4 = tf.Variable(tf.truncated_normal(shape=(800, 43), mean = mu, stddev = sigma), name="W4")
    b4 = tf.Variable(tf.zeros(43), name="b4")    
    logits = tf.add(tf.matmul(x, W4), b4)
    
    #################
    #################
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 5: Fully Connected. Input = 0. Output = 84.
    W5 = tf.Variable(tf.truncated_normal(shape=(800, 32), mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(32)) 
    x = tf.add(tf.matmul(x, W5), b5)
    
    return logits



tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, n_classes)

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# EPOCH 5: 0.926 & LeNet2
EPOCHS = 25
BATCH_SIZE = 48

rate = 0.0009

logits = LeNet2(x)
#logits = LeNetORIG(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_aug)
    
    print("Training...")
    print()
    testAccuracyPlot = []
    for i in range(EPOCHS):
        # shuffle because we're using small batches
        X_train, y_train = shuffle(X_train_aug, y_train_aug)
        #X_train, y_train = X_train_aug, y_train_aug
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        testAccuracy = evaluate(X_test, y_test)
        testAccuracyPlot.append(testAccuracy)
        print("EPOCH {} ...".format(i+1))
        print("Test Accuracy = {:.3f}".format(testAccuracy))
        print()
        if i+1 == EPOCHS: 
            validationAccuracy = evaluate(X_valid, y_valid)
            print("Validation Accuracy = {:.3f}".format(validationAccuracy))
    saver.save(sess, 'c:\\Data\\lenet')
    print("Model saved")

plt.plot(testAccuracyPlot)
plt.title("Test Accuracy")
plt.show()

validationAccuracy = evaluate(X_valid, y_valid)

###
signnames = pd.read_csv('signnames.csv')
#signnames = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint','i8'), ('mysring','S55')], delimiter=',')

def PlotImages(figures, nrows = 1, ncols=1, labels=None):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 14))
    axs = axs.ravel()
    for index, title in zip(range(len(figures)), figures):
        axs[index].imshow(figures[title], plt.gray())
        if(labels != None):
           axs[index].set_title(labels[index])
        else:
            axs[index].set_title(title)
            
        axs[index].set_axis_off()
        
    plt.tight_layout()


# I selected some signs that weren't in the original data set and denote them as -1
myImages = sorted(glob.glob('./images/*.jpg'))
myLabels = np.array([32,3,-1,-1,-1,-1,18,41,9])

    
figures = {}
labels = {}
mySigns = []
index = 0
for image in myImages:
    img = cv2.imread(image)
    mySigns.append(img)
    figures[index] = img
    if myLabels[index] != -1:
        labels[index] = signnames.iloc[myLabels[index]][1]
    else:
        labels[index] = 'unknown'
    index += 1

PlotImages(figures, 3, 3, labels)

np.array(mySigns).shape


"""
make them gray and normalize them
"""

mySigns = np.array(mySigns)
mySigns_gray = np.sum(mySigns/3, axis=3, keepdims=True)
mySignNormalized = mySigns_gray/127.5-1

number_to_stop = 6
figures = {}
labels = {}
for i in range(len(myImages)):
    if myLabels[i] != -1:
        labels[i] = signnames.iloc[myLabels[i]][1]
    else:
        labels[i] = 'unknown'
    figures[i] = mySigns_gray[i].squeeze()
    
PlotImages(figures, 3, 3, labels)



## Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'c:\\Data\\lenet')
    myAccuracy = evaluate(mySignNormalized, myLabels)
    print("My Data Set Accuracy = {:.3f}".format(myAccuracy))

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
oneImage = []
oneLabel = []

for i in range(len(myImages)):
    oneImage.append(mySignNormalized[i])
    oneLabel.append(myLabels[i])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, 'c:\\Data\\lenet')
        myAccuracy = evaluate(oneImage, oneLabel)
        print('Image {}'.format(i+1))
        print("Image Accuracy = {:.3f}".format(myAccuracy))
        print()

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
n = len(myImages)
softmax_logits = tf.nn.softmax(logits)
top5 = tf.nn.top_k(softmax_logits, k=n)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'c:\\Data\\lenet')
    myLogits = sess.run(softmax_logits, feed_dict={x: mySignNormalized, keep_prob: 1.0})
    myBestGuesses = sess.run(top5, feed_dict={x: mySignNormalized, keep_prob: 1.0})

    for i in range(0,n):
        figures = {}
        labels = {}
        
        figures[0] = mySigns[i]
        labels[0] = "Original"
        
        for j in range(5):
            labels[j+1] = 'Guess {} : ({:.0f}%)'.format(j+1, 100*myBestGuesses[0][i][j])
            figures[j+1] = X_valid[np.argwhere(y_valid == myBestGuesses[1][i][j])[0]].squeeze()
            
        PlotImages(figures, 1, 6, labels)



