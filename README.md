#**Behavioral Cloning for Self Driving Car** 

## Installation & Resources

1. Python 3.5 and the following packages:
	* socketio
	* eventlet
	* PIL
	* flask
	* opencv
	* keras
	* tensorflow

2. Udacity SDC simulator [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip), [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip), [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip),  [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

3. image data and steering angle data (produced by Udacity and myself)

## Files and Usage

* `model.py` for training model

* `model.json` model structure saved as json

* `model.h5` model weights saved as h5

* `drive.py` code that takes in image from the simulator and runs it through `model.json` and loaded with weights from `model.h5` and returns a steering angle and a throttle back to the simulator

To use this, start the simulator in Autonomous mode and run `python drive.py model.json` from the terminal.

## Overview
This is the third project in the Udacity Self Driving Car Nanodegree Program. The goal of the project is to use deep learning to train a simulated car to drive on the designated track. 

### Problem Definition
There are two track options in the simulator. The performance of the model is evaluated based on whether it can drive the car successfully around the first track. To get the car to drive well on the second track is considerably more challenging if you are only allowed to use the data collected from the first track.

### Approach
#### Why is behavioral cloning hard?
In this project there is something I find quite interesting. On the Udacity SDCND Slack channel, there are a lot of students who find this project extremely difficult, and the main reason being that:

**MSE is not a good indicator in this project, and without a good indicator, we're literately tuning our model in the dark.**

When we use MSE as the cost function, essentially we're trying to minimize the sum of MSE across the entire dataset. However, a model that minimizes the sum of MSE doesn't guarantee consistent predictions across the entire dataset. It may have made really good predictions for the most part but failed terribly in some cases, like for sharp turns, and this would not reflect on the MSE. 

Machine learning is generally "hard" because when something doesn't work as expected, it's not very easy to pinpoint what's causing the problem (there's a really good article on this [Why is Machine Learning "Hard"](http://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html)). This is more so in deep learning, especially when we don't have a simple indicator such as MSE. If the only reliable way for us to tune our model, is to run the simulator and see for ourselves, then we know we are not formulating the problem well, because this metric isn't really quantifiable (one moral from Andrew Ng is that it is very important to **"establish a single-number evaluation metric for your team to optimize"**).

A lot of people have pointed out that the behavior of the car is really unpredictable in terms of how you train your network. 

#### So what's the solution?
##### Make you have a balance dataset
To make sure the car drives itself well, we need to make sure it drives well in all circumstances. And in terms of MSE, we need to have:
1. **Small MSE overall**—so that the behavior of the machine is close to the behavior of the human 
2. **Small MSE variance**—so that the behavior is consistent throughout

When we drive around the course, there are more times when the road is straight then when there's a turn. If we just feed this driving data directly to the CNN, it may treat those turns as some kind outliers, while those turns are equally if not more important the straight data. Having a balance dataset is like telling the algorithm that all data should be treated equally. We can use this as a workaround so that our metric MSE is now again meaningful.
 
### Data Collection
The Udacity simulator will be used to generate data of good driving behavior, which includes the front perspective images (including images captured by three cameras: center, left, right), steering angle, throttle, break and speed. To simplify our approach, I will only use image data and the steering angle.

I generated 133,569 observations using the simluator as the training data, and use the dataset provided by Udacity (of 24,108 observations) as the validation data. The fact that these two datasets are not drawn from the same distribution can be a good indicator whether the model generalizes well.

### Data Preprocessing
The following steps are taken for data preprocessing:
1. **Create a balance dataset**
2. **Cropping**—the sky and the front cover are cropped out as they don't provide any useful information
3. **Convert the image into the HLS color space**—it's easier for the neural network to extract good feature representation from HLS images
4. **Normalization**

#### Create a balance dataset
As said earlier, it is crucial to make sure that we have a balance dataset before training. A quick look at the steering angle histogram and we can see that more of the data are near the 0.

![Alt text](./st_hist.png)

The peaks at around $\pm 0.2$ and the duplicated pattern come from my manipulation of the left/camera images. The images are used to simulate cars driving close to the edges of the lane. The cameras are set up to form angles of $\pm 0.25$ with the center camera so I made a slight adjustment to them so that the steering angles of the left cameras images are compensated by $+0.2$ and the right by $-0.2$.

I put the angles in 20 bins and apply a threshold of 2000 so that no bins can have more than 2000 entries. And the result is a far more balanced dataset:

![Alt text](./st_hist2.png)

We can see that the sharp turns still don't have a lot of records and the model can definitely benefit from more data. But for our purpose here this is good enough.

#### Cropping
The sky and the front cover of the car are cropped out as they don't provide any useful information. The model still works without the cropping but it's more efficient this way and reduces a lot the training time.

#### HLS color space
I've tried using the RGB color space for this but it didn't work out. It may take a more complicated autoencoder for this, and if simple HLS conversion can do that job than why not?

#### Normalization
See the Model Building section.

### Model Building
In this project I use Python keras package with tensorflow backend to build the convolution neural network. I've tried re-implementing the NVIDIA end-to-end paper model, however for some reason it doesn't work as well as the Yadav model, which is VGG-styled model architected by Vivek Yadav (check out his awesome [medium post!](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ws7qx41g7)). I've tweaked the model slightly and employed different padding and dropout strategies, though.

Here's the architecture of the model:

* Input Layer—1@35x160x3
* Convolutional Layer 1—3@35x160 (`k=1`, `p='valid'`)
* Convolutional Layer 2—32@33x158 (`k=3`, `p='valid'`)
* Convolutional Layer 3—32@31x156 (`k=3`, `p='valid'`)
	* Maxpooling—32@16x78
	* ELU—32@16x78
* Convolutional Layer 4—64@14x76 (`k=3`, `p='valid'`)
* Convolutional Layer 5—64@12x74 (`k=3`, `p='valid'`)
	* Maxpooling—64@6x37
	* ELU—64@6x37
* Convolutional Layer 6—, 128@4x35 (`k=3`, `p='valid'`)
* Convolutional Layer 7—128@2x33 (`k=3`, `p='valid'`)
	* Maxpooling—128@1x17
	* ELU—128@1x17
* Flatten Layer —2176
* Fully Connected Layer 1—512
	* ELU—512
* Fully Connected Layer 2—64
	* ELU—64
* Fully Connected Layer 3—16 
	* LeakyReLU—16
* Output Layer—1

A generator is used during the training stage mainly for the purpose of saving memory (use `numpy` broadcasting for image normalization is extremely space costly!)