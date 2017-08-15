### SDC-term1
    
    Tung Thanh Le
    ttungl at gmail dot com
   
**Traffic Sign Recognition** 

---
##### Project description:
+ Use `deep neural networks`(DNN) and `convolutional neural networks`(CNN) to build traffic sign recognition. Specifically, we train a deep neural network model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and [traffic signs downloaded from internet](https://github.com/ttungl/SDC-term1-Traffic-Sign-Classifier/tree/master/Traffic%20Sign%20Classifier/test_german_trafficsign).
+ My implementation can be downloaded from the following links [[ipynb]](https://github.com/ttungl/SDC-term1-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier_Final.ipynb) or [[html]](https://github.com/ttungl/SDC-term1-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier_Final.html).
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

[image1]: ./images_output/images_input.png "Input images"
[image2]: ./images_output/occurrences_dataset.png "Occurrences"
[image3]: ./images_output/images_input_grayscale.png "Grayscale"
[image4]: ./images_output/newimages_grayscale_with_model.png "New images processed"
[image5]: ./images_output/newimages_predict_signs_with_model.png "Predicted Signs"
[image6]: ./images_output/softmax_0.png "Original new image"
[image7]: ./images_output/softmax_1.png "Grayscale new image"
[image71]: ./images_output/softmax_2.png "new image"
[image72]: ./images_output/softmax_3.png "new image"
[image73]: ./images_output/softmax_4.png "new image"
[image74]: ./images_output/softmax_5.png "new image"
[image75]: ./images_output/softmax_6.png "new image"
[image76]: ./images_output/softmax_7.png "new image"
[image77]: ./images_output/softmax_8.png "new image"
[image78]: ./images_output/softmax_9.png "new image"

<!-- conv1 layer -->

[image8]: ./images_output/conv1_img0.png "Input images"

[image9]: ./images_output/conv1_img1.png "Input images"

[image10]: ./images_output/conv1_img2.png "Input images"

[image11]: ./images_output/conv1_img3.png "Input images"

[image12]: ./images_output/conv1_img4.png "Input images"

[image13]: ./images_output/conv1_img5.png "Input images"

[image14]: ./images_output/conv1_img6.png "Input images"

[image15]: ./images_output/conv1_img7.png "Input images"

[image16]: ./images_output/conv1_img8.png "Input images"

[image17]: ./images_output/conv1_img9.png "Input images"

<!-- conv1 maxpool layer -->

[image18]: ./images_output/conv1_maxpool_img0.png "Input images"

[image19]: ./images_output/conv1_maxpool_img1.png "Input images"

[image20]: ./images_output/conv1_maxpool_img2.png "Input images"
[image21]: ./images_output/conv1_maxpool_img3.png "Input images"
[image22]: ./images_output/conv1_maxpool_img4.png "Input images"
[image23]: ./images_output/conv1_maxpool_img5.png "Input images"
[image24]: ./images_output/conv1_maxpool_img6.png "Input images"
[image25]: ./images_output/conv1_maxpool_img7.png "Input images"
[image26]: ./images_output/conv1_maxpool_img8.png "Input images"
[image27]: ./images_output/conv1_maxpool_img9.png "Input images"
<!-- conv2 layer -->

[image28]: ./images_output/conv2_img0.png "Input images"
[image29]: ./images_output/conv2_img1.png "Input images"
[image30]: ./images_output/conv2_img2.png "Input images"
[image31]: ./images_output/conv2_img3.png "Input images"
[image32]: ./images_output/conv2_img4.png "Input images"
[image33]: ./images_output/conv2_img5.png "Input images"
[image34]: ./images_output/conv2_img6.png "Input images"
[image35]: ./images_output/conv2_img7.png "Input images"
[image36]: ./images_output/conv2_img8.png "Input images"
[image37]: ./images_output/conv2_img9.png "Input images"
<!-- conv2 maxpool layer -->

[image38]: ./images_output/conv2_maxpool_img0.png "Input images"
[image39]: ./images_output/conv2_maxpool_img1.png "Input images"
[image40]: ./images_output/conv2_maxpool_img2.png "Input images"
[image41]: ./images_output/conv2_maxpool_img3.png "Input images"
[image42]: ./images_output/conv2_maxpool_img4.png "Input images"
[image43]: ./images_output/conv2_maxpool_img5.png "Input images"
[image44]: ./images_output/conv2_maxpool_img6.png "Input images"
[image45]: ./images_output/conv2_maxpool_img7.png "Input images"
[image46]: ./images_output/conv2_maxpool_img8.png "Input images"
[image47]: ./images_output/conv2_maxpool_img9.png "Input images"
<!-- ![alt text][image8] --> 

This implementation addressed each point of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as below. 

## Dataset Exploration

#### Dataset Summary

* The size of training dataset is `27839` images, validation set size is `4410` images, and test set size is `12630` images. I use numpy library to get the shape of images, `(32, 32, 3)`. The number of classes is `43`. 

#### Exploratory Visualization

* First, I plot `43` images in the training dataset as in `Figure 1`. The data shows that the input images are random in the set in terms of the classes. 
![alt text][image1]

Figure 1: Input training dataset. 

Then, I get the statistical analysis of the dataset in terms of the number of occurrences in each class as in `Figure 2`. 
![alt text][image2] 

Figure 2: Number of occurrences for each class. 

The plot shows that the amount of examples in each class is imbalanced. The largest amount of examples are classes `1, 2`, which are around `1600` examples for each class. 

## Design and Test a Model Architecture

#### Preprocessing techniques

* As recommended, I use a quick way to approximately normalize data, `(pixel-128.)/128.`. Then, grayscale is used to convert the RGB image into GRAY image, using OpenCV library `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`. After that, I reshape the images to the size of `(32,32)`. The result of this process is as follows.
![alt text][image3]

Figure 3: Grayscale images processed. 

There is a technique called [spatial transformer](https://github.com/tensorflow/models/tree/master/transformer), which allows the spatial manipulation of image within the network. This technique helps eliminate the white noise of the input images. I think this can be used later to improve the quality of the input image in my implementation. 

* Update: The [brightness augmentation](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc) is also a robust alternative to improve the performance of classifications. 

#### Model Architecture

* I modified the LeNet Architecture by adding dropout probabilities between fully-connected layers.

	`One layer -> dropout -> another layer`

				`1.0 -> 1.0`
				`0.2 -> 0.2`
				`0.4 -> 0.4 (x)`
				`-0.3 -> -0.3 (x)`

* It takes those activations randomly, for every example you train on your network, set half of them to `zero` that are flowing through the network, just destroy it and then randomly again.

				`1.0 -> 1.0 (x)`
				`0.2 -> 0.2`
				`0.4 -> 0.4`
				`-0.3 -> -0.3 (x)`				

* The purpose of this dropout is that the network can never rely on any given activation to be present because they maybe squashed at any given moment. It's forced to learn a redundance for everything. This is to ensure at least some of the information remains. In practice, it makes more robust and prevents overfitting. 

* Note that, if dropout probabilities method does not work, probably you would need to using a bigger network.

* I proposed two network architectures, the first architecture `LeNetUpgrade` uses ReLUs activations, and second one `LeNetUpgrade_tanh` uses tanh activations. 

**`LeNetUpgrade` Architecture

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image   				| 
| Convolution 2d     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU 			| activation function		 		|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 2d	| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU 			| activation function		 		|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten		| 5x5x16 -> 400x1				|
| Fully connected	| 400x1 -> 120x1				|
| RELU 			| activation function		 		|
| Dropout		| keep_prob=0.5				 	|
| Fully connected	| 120x1 -> 84x1					|
| RELU 			| activation function		 		|
| Dropout		| keep_prob=0.5				 	|
| Softmax		| 84x1 -> 43x1					|

---

(I used this model)
**`LeNetUpgrade_tanh` Architecture 

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image   				| 
| Convolution 2d     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Bias added		| add bias to convolution output 		|
| tanh 			| activation function				|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 2d	| 1x1 stride, valid padding, outputs 10x10x16   |
| Bias added		| add bias to convolution output 		|
| tanh 			| activation function		 		|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten		| 5x5x16 -> 400x1				|
| Fully connected	| 400x1 -> 120x1				|
| tanh 			| activation function		 		|
| Dropout		| keep_prob=0.5				 	|
| Fully connected	| 120x1 -> 84x1					|
| tanh 			| activation function		 		|
| Dropout		| keep_prob=0.5				 	|
| Softmax		| 84x1 -> 43x1					|

---

#### Model Training

* The model was trained by using the following parameters:

| Parameter      	|  Setting	| 
|:---------------------:|:-------------:| 
| EPOCHS         	|  	50	| 
| BATCH_SIZE    	|  	128	| 
| LEARNING_RATE  	|  	0.001	|
| KEEP_PROB        	|  	0.5	|
| beta (REGULARIZATION)	|  	0.001	|

* To minimize the loss, I use one-hot encoded and softmax cross entropy for the logits. Then, I use the L2-regularization to prevent overfitting by using `newloss = loss + beta*regularization`. `beta` is set at `0.001`. At first, I pretended to use SGD with learning rate decay but the ADAM optimizer seems to be a good option as it's simple and performs well without additional hyperparameters. So, I used ADAM optimizer for the training process. 

* The `BATCH_SIZE` is `128`, and the number of `EPOCHS` is `50`. `LEARNING_RATE` is set at `0.001` and keep_prob is `0.5`. 

#### Solution Approach

	EPOCH 50 ...
	Train Accuracy = 0.990
	Validation Accuracy = 0.937
	Test Accuracy = 0.921

* After `50` epochs, my validation accuracy is `93.7%`, and train accuracy is `99.0%`, and test accuracy is `92.1%`.  

* I observed that, when using both `LeNetUpgrade` and `LeNetUpgrade_tanh`, no significant improvement was made. Just keep the `KEEP_PROB` at `0.5` is fairly reasonable for half of data set to zero, for every example. 

## Test a Model on New Images

#### Acquiring New Images

* The new images are downloaded from the internet and then preprocessed them as in Figure 4. This time, instead using OpenCV, I use the different techniques to grayscale, normalize, and standardize the new images as follows. 

	+ For grayscale images, I import `rgb2gray` library from `skimage.color`. 

	+ For normalize scale, I use [min-max scaling method](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling). 

	+ For standardize, I use the preprocessing approach from [this link](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html) to obtain the zero-center, and then normalize them.

After the process, the new images show as below.

![alt text][image4]

Figure 4: Grayscale new images processed. 

#### Performance on New Images

I use the proposed neural network model to test the new images downloaded from internet. The result shows in Figure 5 as follows.

![alt text][image5]

Figure 5: Predicted signs from 10 new images of the model. Note, A# is the actual sign; P# is the predicted sign.

* From the result, we see that the model works properly with 70% accuracy. There are three signs that the model predicted incorrectly per 10 signs. The top left sign is the sign of `be aware of ice/snow` has been covered mostly by snow, so it's hard to recognize this image, the model predicted it as a `roundabout mandatory` (`40`). The second image from the top left sign is the sign of `children crossing`, however, this image has been distorted after preprocessing. I observed that this image originally was too wide, so after reshaping it, the sign is distorted, therefore, the model found difficult to recognize it. It predicted this sign as a `speed limit 80km/h`. The second image from the bottom left is `speed limit 100 km/h`. It's blurred and there are some obstacles in front of it, so the model failed to recognize this one. Other than that, all the new images are clear to be recognized correctly by the model. 

The prediction result of my neural network model for new images:

| New Image      		|  Prediction									| 
|:---------------------:|:---------------------------------------------:|	 
| Be Aware of Ice/Snow  | General caution    							|  
| Children crossing   	| Right-of-way at the next intersection			|
| No entry 			    | No entry										|
| Roundabout mandatory 	| Roundabout mandatory							|
| Slippery road 		| Slippery road									|
| Speed limit 70km/h 	| Speed limit 70km/h 							|
| Speed limit 100km/h 	| No passing for vehicles over 3.5 metric tons 	|
| Speed limit 60km/h 	| Speed limit 60km/h 							|
| Stop 			   		| Stop											|
| Turn left ahead 		| Turn left ahead								|


* Note: As mentioned, if using the [spatial transformer](https://github.com/tensorflow/models/tree/master/transformer) to preprocess the images properly, the prediction accuracy will be improved for classifications.

#### Model Certainty - Softmax Probabilities

The softmax probabilities are visualized as below. 

![alt text][image6] ![alt text][image7]

![alt text][image71] ![alt text][image72]

![alt text][image73] ![alt text][image74]

![alt text][image75] ![alt text][image76]

![alt text][image77] ![alt text][image78]

## Visualize Layers of the neural network

I export the images in each layers as follows.

#### Convolutional layer 1:

Be Aware of Ice/Snow 

![alt text][image8]

Children crossing  

![alt text][image9]

No entry

![alt text][image10]

Roundabout mandatory 

![alt text][image11]

Slippery road 

![alt text][image12]

Speed limit 70km/h 

![alt text][image13]

Speed limit 100km/h 

![alt text][image14]

Speed limit 60km/h

![alt text][image15]

Stop

![alt text][image16]

Turn left ahead 

![alt text][image17]

#### Convolutional layer 1 Max pooling:

Be Aware of Ice/Snow 

![alt text][image18]

Children crossing

![alt text][image19]

No entry

![alt text][image20]

Roundabout mandatory

![alt text][image21]

Slippery road 

![alt text][image22]

Speed limit 70km/h

![alt text][image23]

Speed limit 100km/h

![alt text][image24]

Speed limit 60km/h

![alt text][image25]

Stop

![alt text][image26]

Turn left ahead 

![alt text][image27] 

#### Convolutional layer 2:

Be Aware of Ice/Snow 

![alt text][image28]

Children crossing

![alt text][image29]

No entry

![alt text][image30]

Roundabout mandatory

![alt text][image31]

Slippery road

![alt text][image32]

Speed limit 70km/h

![alt text][image33]

Speed limit 100km/h

![alt text][image34]

Speed limit 60km/h

![alt text][image35]

Stop

![alt text][image36]

Turn left ahead 

![alt text][image37] 

#### Convolutional layer 2 Max pooling:

Be Aware of Ice/Snow 

![alt text][image38]

Children crossing

![alt text][image39]

No entry

![alt text][image40]

Roundabout mandatory

![alt text][image41]

Slippery road

![alt text][image42]

Speed limit 70km/h

![alt text][image43]

Speed limit 100km/h

![alt text][image44]

Speed limit 60km/h

![alt text][image45]

Stop

![alt text][image46]

Turn left ahead

![alt text][image47]
