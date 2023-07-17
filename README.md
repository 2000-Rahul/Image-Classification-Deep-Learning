## Image Classification Model Using Deep Learning 
**Image Classification** <br>
The task of identifying what an image represents is called image classification. An image classification model is trained to recognize various classes of images. <br>
**Deep Learning** <br>
It is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain although far from matching its ability—allowing it to “learn” from large amounts of data.


## Model Overview
- This model is train to recognize photos representing two different human facial expressions : Happy and sad. <br>
- To train this model data is collected from google url of Happy and Sad people with facial expressions.
- Images  with these four extensions are only included to train the model - ["jpg","jpeg","bmp","png"]
- All these images are stored inside Data folder, happy images are stored inside "happy' name folder, sad images are stored inside 'sad' name folder.
- To train this model Convolutional Neural Netwrok (CNN) is used.

## CNN - Convolutional Neural Netwrok
- A CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data.There are other types of neural networks in deep learning, but for identifying and recognizing objects, CNNs are the network architecture of choice. This makes them highly suitable for computer vision  tasks and for applications where object recognition is vital, such as self-driving cars and facial recognition.

## Model Building Steps

**1. Insatall Required Dependencies**  
- Install dependencies with pip command inside jupyter notebook
- pip install tensorflow
- pip install opencv-python matplotlib

**2.Import necessary libraries for project inside Python**
- Tensorflow, os module, cv2, imghdr, matplotlib

**3.Remove Dodgy images** <br>
- With the help of OS module read single image and plot it using matplotlib library.
- Now, a function is built to select only selected images from the data folder.
- Images which have these extensions are filtered -  ("jpg","jpeg","bmp","png") from the image dataset by the help of above function.

**4.Load Data** <br>
- First, we will use high-level Keras preprocessing utilities (such as tf.keras.utils.image_dataset_from_directory)  to read a directory of images on disk.
- Create an Iterator, By using the created dataset to make an Iterator instance to iterate through the dataset.
- By creating into iterator two batches are formed, first contain actual images of dataset, and second contain labels of these images.

**5. Scaling of data** <br>
- image scaling refers to the resizing of a digital image.
- All images are resized in (256,256,3) shape.
  
**6. Splitting Data** <br> 

 - Image dataset is divided into three portions - training dataset(4 batches) , validation dataset(1 batch) , testing dataset(1 batch)

**7. Building deep learning model** <br>

In this step, you will build the architecture of the classification convolutional neural network. <br>

from tensorflow.keras.models import Sequential # functional Api can also be used <br>
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

- we will be building the model in a sequential way. This means that you will layer after layer. <br>
  we add 3 convolutional layers and 3 maxpooling layers.
  In the convolutional layers, multiple filters are applied to the image to extract different features. <br>

- Input-shape: The image given should be of the shape (224,224,3).

- Filters: The number of filters that the convolutional layer will learn.

- Kernel_size: specifies the width and height of the 2D convolution window.

- Activation: This is more of a convenience argument. Here, we specify which activation function will be applied after the convolutional layers. We will apply the ReLU activation function.
  
- Max Pooling : Pooling is used to reduce the dimensionality of images by reducing the number of pixels in the output of the previous convolutional layer.

- Now the convolutional base is created. To be able to generate a prediction, we have to flatten the output of the convolutional base.

- Add the dense layers. The dense layers feeds the output of the convolutional base to its neurons. <br>

Arguments: Units: Number of neurons , Activation function: Relu <br>

The Relu activation function speeds up training since the gradient computation is very simple (0 or 1). This also implies that negative values are not passed or “activated” on to the next layer. This makes that only a certain number of neurons are activated .

**8. Loss function & optimizer**

- In this deep learning model we use adam optimizer and Binary Cross Entrop loss function of keras.

- The basic idea behind Adam optimization is to adjust the learning rate adaptively for each parameter in the model based on the history of gradients calculated for that parameter. This helps the optimizer converge faster and more accurately than fixed learning rate methods like stochastic gradient descent (SGD)

- Binary Cross Entropy <br>

The Binary Cross entropy will calculate the cross-entropy loss between the predicted classes and the true classes. By default, the sum_over_batch_size reduction is used. This means that the loss will return the average of the per-sample losses in the batch. <br>

- Accuracy classifer is  used in this deep learning model,  which predict how well oue model is classifying images.

**9. Training Model**
- Model is train with training data with the help of model.fit() function
- 
- tf.keras.callbacks.TensorBoard() <br>

 we use tensorboard tool of tensorflow which is a visualization tool provided with TensorFlow. This callback logs events for TensorBoard, including: Metrics summary plots.     Training graph visualization.
 
-Two line graphs are plotted which shows Model accuracy and Model loss 

**10. Evaluate Model Performance**
-from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
we calculate precision, recall, binary accuracy of model and out put of all of these three are exact 1.

**11. Evaluating performance of model on Unknown images**

- we evaluate our model with different images of five chosen extension and model is classifying the happy and sad images correctly.

 ![image](https://github.com/2000-Rahul/Image-Classification-Deep-Learning/assets/136818857/2de40068-4109-4dfa-a8a3-bc6c748a89ea)

![image](https://github.com/2000-Rahul/Image-Classification-Deep-Learning/assets/136818857/3a6a0d87-d684-4f5f-b44f-f8774f97e7ce)











