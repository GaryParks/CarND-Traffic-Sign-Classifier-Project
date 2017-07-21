**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visual.png "Visualization"
[image2]: ./images/normal.png "Normalization"
[image4]: ./new-traffic-signs-data/03_Speed-limit-(60kph).jpg "Traffic Sign 1"
[image5]: ./new-traffic-signs-data/09_No-passing.jpg "Traffic Sign 2"
[image6]: ./new-traffic-signs-data/17_No-entry.jpg "Traffic Sign 3"
[image7]: ./new-traffic-signs-data/32_End-of-all-speed-and-passing-limits.jpg "Traffic Sign 4"
[image8]: ./new-traffic-signs-data/33_Turn-right-ahead.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickles library to load the data set and used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle the dataset.  I then proceeded with normalizing the data, which can be found in code block 5.  I had tried using grayscaling and histogram but I had mixed reults when validating the model, so I kept the image processing to a minimum. 

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution1 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU1					| Outputs 28x28x6						     	|
| Max pooling1	      	| 2x2 stride, outputs 14x14x6				    |
| Convolution2 5x5	    | 1x1 stride, same padding, outputs 10x10x16   	|
| RELU2		            | Outputs 10x10x16        						|
| Max pooling2			| 2x2 stride, outputs 5x5x16            		|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120									|
| Fully connected		| Outputs 84									|
| Fully connected		| Outputs 43									|
|                   	|                 								|
 
This is the same LeNet that is from the LeNet Lab.  I just modified the input to 32, 32, 3 to accept color images, and the output to match the 43 classes of the traffic signs.


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an atom optimizer from the LeNet Lab using the following parameters: 

Epochs: 20 \
batch size: 128 \
mu: 0 \
sigma: 0.1\
learning rate: 0.0025 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.3%
* test set accuracy of 91.6%

I used the same LeNet structure that we used in the lab.
I first modified the input to 32, 32, 3 to accept color images, and the output to match the 43 classes of the traffic signs.

I initially tried using grayscale and histogram to process the data sets but I could not reach the required 93% validation accuracy.  I then decided to keep it simple and just normalize the data.  I also ended up changes to 20 EPOCHs and adjusting the learning rate to 0.0025 to achieve 94.3% validation accuracy.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All of the signs were clearly visible.  If anything, I would have thought that the third sign could be difficult due to the angle from which the photo was taken from.
The first three signs also have some scenery in the background which could affect the classification.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Actual		                      |     Prediction	        				| 
|:-----------------------------------:|:---------------------------------------:| 
| Speed limit (60km/h)           	  | Speed limit (60km/h)   					| 
| No passing     			          | Priority road 							|
| No entry					          | No entry								|
| End of all speed and passing limits | End of all speed and passing limits		|
| Turn right ahead		              | Speed limit (100km/h)    				|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. I am fairly surprised with the results.  The third sign may be due to the background, but I cannot think of a reason why the model would fail on the fifth image.  I may need to include more of each of these signs in the dataset to improve the model.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 60 kph sign (probability of 0.999993), and the image does contain a 60 kph sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999993      		| Speed limit (60km/h)  						| 
| 0.000006     			| Speed limit (50km/h) 							|
| 0.000000				| Speed limit (30km/h)							|
| 0.000000      		| Wild animals crossing					 		|
| 0.000000			    | Speed limit (80km/h)     						|


For the second image, the model is relatively sure that this is a priority road sign (probability of 0.999995), and the image does not contain a priority road  sign, nor was it in the top five softmax priorites. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999995      		| Priority road   								| 
| 0.000002     			| Traffic signals 								|
| 0.000001				| Right-of-way at the next intersection			|
| 0.000000      		| Roundabout mandatory					 		|
| 0.000000			    | End of no passing      						| 


For the third image, the model is relatively sure that this is a no entry sign (probability of 1.000000), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000      		| No entry   									| 
| 0.000000     			| Turn left ahead 								|
| 0.000000				| Dangerous curve to the right					|
| 0.000000      		| Slippery road					 				|
| 0.000000			    | No passing for vehicles over 3.5 metric tons  |


For the forth image, the model is relatively sure that this is a end of all speed and passing limits sign (probability of 0.990511), and the image does contain that sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.990511      		| End of all speed and passing limits   		| 
| 0.006279     			| End of speed limit (80km/h) 					|
| 0.003014				| End of no passing								|
| 0.000129      		| Roundabout mandatory					 		|
| 0.000047			    | Right-of-way at the next intersection      	|


For the fifth image, the model is relatively sure that this is a 100 kph sign (probability of 0.661951), and the image does not contain a 100 kph sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.661951      		| Speed limit (100km/h)   						| 
| 0.231696     			| Speed limit (30km/h)							|
| 0.040848				| General caution								|
| 0.031577     			| Roundabout mandatory					 		|
| 0.024766			    | Right-of-way at the next intersection    		|