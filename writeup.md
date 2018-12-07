# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/visualization.jpg "Visualization"
[image2]: ./results/grayscale.jpg "Grayscaling"
[image3]: ./results/visualization2.jpg "Random images"
[image4]: ./results/web_images.jpg "Web Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/ashwinmr/Ud_CarND_P3/blob/master/main.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy and the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

Here are a random set of images:
![alt text][image3]

Here is a histogram showing the frequency of the labels
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I first tried implementing LeNet as-is but with 3 color layers for the images.

I noticed that my training accuracy was much higher than my validation accuracy
which meant that I was overfitting the data. So I decided to reduce the features
by converting images to grayscale. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so that it has mean zero and equal variance to improve learning

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		| output 120        			      	     	|
| RELU                  |                                               |
| Fully connected		| output 84        			      		        |
| RELU                  |                                               |
| Fully connected		| output 43        			      		        |
| Softmax				|            									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I reduced the batch size to 50 to improve accuracy
I also increased the number of epochs to 12 to get more learning and higher accuracy

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.937
* test set accuracy of 0.906

If an iterative approach was chosen:
* I first tried the LaNet architecture since I was familiar with it and it was used to solve a similar problem
* The first problem I faced was that LaNet was made for grayscale images. I modified the placeholder to accept color images. This caused poor accuracy of my training and validation sets. This indicated that my model was underfitting.
* I increased the complexity of the model by multiplying all output layers by 3 to compensate for the color images. This caused my training accuracy to become much higher than validation accuracy. This indicated that I was now overfitting my training set.
* This is when I chose to convert the images to grayscale to reduce the number of features to learn and the idea that colors are not important to differentiate traffic signs. This improved my training and validation accuracy therefore my model was working well.
* Finally I tuned the batch size and epochs to achieve the target accuracy. I reduced batch size and increased epochs.

* A convolution layer works well because the traffic sign can be at any position and small features need to be picked out first. Dropout would also help to make the detection more robust. 
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that I found on the web:

![alt text][image4]

The first image should have been easy to classify since it is bright and clear, but it ended up being hard because it looks similar to other speed limit signs. 
The fourth image should be hard to classify since it is not very clear. 
The fifth image should be hard to classify since it has a different perspective.
The other images should be easy to classify.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Speed limit (20km/h) 							| 
| Turn right ahead      | Turn right ahead  							|
| Road work			    | Road work										|
| Double curve	        | Slippery road					 				|
| Pedestrians		    | Pedestrians       							|
| Stop                  | Stop                                          |


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 67%. This compares worse than the accuracy on the test set of 90%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the third image, the model is relatively sure that this is a Road work sign (probability of 0.99), and the image does contain a ROad work sign. The top five soft max probabilities were

| Probability         	    |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| .99898        			| Road work  						            | 
| .00096     				| No passing for vehicles over 3.5 metric tons 	|
| .00003					| Bicycles crossing							    |
| .00001	      			| Wild animals crossing		 				    |
| .00000				    | Speed limit (80km/h)     					    |

Other probabilities and predictions are present in the jupyter output.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
