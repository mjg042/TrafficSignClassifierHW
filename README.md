# TrafficSignClassifierHW

# Self Driving Car Project 2 - Traffic Sign Classifier
This doc describes the work I did for the Udacity Self Driving Car Traffic Sign Classification Project.

The Jupyter notebook Traffic_Sign_Classifier contains all the code and the results of the 
execution of that code.

The challenge of this project is to train a convolutional neural network model to classify German traffic signs. Students were given 

[//]: # (Image References)

[image1]: ./origImages/german_1.jpg "Road Work"
[image2]: ./origImages/speed50.jpg "Speed Limit 50kph"

![Road Work][image1]

and this:

![Speed Limit 50kph][image2]

Moreover, the project solution had to process a video taken from cars driving on the highway.

## Processing Pipeline
To accomplish this task, I built an image processing pipeline, which consisted of the following steps.

* Read train, test and validation image sets
* Perform some basic analysis of the images
* Explore the images
* Convert all images to grayscale
* Augment the images by transforming the original images in various ways
* Normalize the images
* Create a CNN model architecture
* Train, test and validate model
* Test model on new & unseen images
* Analyze performance of the model

To get started, Udacity kindly gave students a number of helper functions. Students then had to modify them to improve them. For example, a number of parameters had to be tweaked to transform the images properly. The final parameterization can be found in the P1.ipynb notebook. 

The biggest hurdle to overcome by students was to use the Hough lines properly. The Hough transformation produces an array of line segments, i.e. every line the algorithm finds in the image. There can be *many* such lines. The tricky part is to find the line segments that represent the lane lines, "paste" those together somehow, and extrapolate them to the proper locations in the image. My function to do this is called `draw_lines`. 

## Shortcomings
In general my solution performed well for the two basic cases
* solidWhiteRight.mp4
* solidYellowLeft.mp4

There was a lot of jitter in the computed lines indicating that my code frequently used the wrong Hough lines, e.g. lines that were not part of the lane lines. The simple modifications I made to track uphill and downhill improved lane line tracking in the solidYellowLeft problem quite a bit

My solution performed *very* poorly for the challenge problem (challenge.mp4). I did not give myself enough time to investigate why, but I suspect it was due to the different image sizes. My solution used too many hardcoded parameters for the image sizes used in the previous two examples. This led to crazy lane line identification that would confuse any self-driving car.


## Possible Improvements

There are a number of possible improvements to make.
1. Linear regression to determine best possible line from a set of line segments
2. Non-linear fitting of curving lines to account for cars travelling around corners
3. Tracking lanes from frame-to-frame to make it easier to determine

&nbsp;&nbsp;&nbsp;&nbsp; * lanes from frame-to-frame

&nbsp;&nbsp;&nbsp;&nbsp; * if car is crossing lanes

4. Defining ranges of possible slopes and intercepts to avoid spurious, i.e. jittery, lines
5. Smoothing slopes and intercepts from one frame to the next. Cars are not generally swerving around the road.
6. More general parameterization for varying image size.







