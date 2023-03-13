# MNIST Dataset Classification 
> ~A standard (non-convolution based) neural network to classify the MNIST dataset.

The MNIST Database contains gray-scale images of 28x28 dimension where each image represents a handwritten digit which the network has to identify.

## Step 1 : Setting up the database 

### Downloading and Transforming the database : 

We need to download the MNIST Dataset and Transform it to Tensors which we are going to input into the model. This is achieved by :

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/0fa674e4325acf4e82ea8513c948062677d04baf/MNIST%20Classification%20Model..py#L8-L9

> 'train' dataset represents our Training dataset and 'test' dataset represents the Testing dataset.

### Getting to know the dataset better : 

To know about number of samples given in dataset, we can simply use :

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/0fa674e4325acf4e82ea8513c948062677d04baf/MNIST%20Classification%20Model..py#L10-L11

To see example of images in training dataset, We can use :

```py
image,label = test[0] #to display the first image in test dataset along with its corresponding number
plt.imshow(image.numpy().squeeze(), cmap='gray_r');
print("\nThe Number is : " ,label,"\n")
```
### Deciding on whether to use batches or not :

The accuracy of the estimate and the possibility that the weights of the network will be changed in a way that enhances the model's performance go up with the number of training examples used. 

A noisy estimate is produced as a result of smaller batch size, which leads to noisy updates to the model, such as several updates with potentially very different estimates of the error gradient. However, these noisy updates sometimes lead to a more robust model and definately contribute to a faster learning.

Various Types of Gradient Descents :
1.   Batch Gradient Descent : The whole dataset is treated as one batch
2.   Stochastic Gradient Descent : Batch size is set to one example.
3.   Minibatch Gradient Descent : Batch size is set to somewhere in between one and total number of examples in the training dataset.

Given that we have quite a large database, we will not take batch size to be equivalent to the whole dataset.

Smaller batch sizes also give us certain benifits such as :

1. Lower generalization error.
2. Easiness in fitting one batch of training data in memory.

We will use mini-batch gradient descent so that we update our parameters frequently as well as we can use vectorized implementation for faster computations.

A batch size of maybe 30 examples would be suitable. 

We would use dataloader for randomly breaking our datasets into small batches :

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/0fa674e4325acf4e82ea8513c948062677d04baf/MNIST%20Classification%20Model..py#L13

## Step 2 : Creating the neural network

### Deciding on Number of Hidden Layers and neurons :

This is a topic of very elaborate discussion but to make it easier, The discussions on : [AI FAQs](http://www.faqs.org/faqs/ai-faq/neural-nets/part3/) were followed in making this model. Thus, The number of hidden layers were decided to be one and the number of hidden nodes in the layer would be 490 (Considering the thumb rule as : The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.)

The input nodes are 784 as a result of 28 x 28 (Number of square pixels in each image), While the Output layer is 10, one for each digit (0 to 9)

This is implemented as :

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/0fa674e4325acf4e82ea8513c948062677d04baf/MNIST%20Classification%20Model..py#L16-L18

