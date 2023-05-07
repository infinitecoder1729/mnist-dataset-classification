# MNIST Dataset Classification 

> ~A standard (non-convolution based) neural network to classify the MNIST dataset.

# Table of contents

- [Step 1 : Setting up the database](#step-1--setting-up-the-database)
  - [Downloading and Transforming the database :](#downloading-and-transforming-the-database-)
  - [Getting to know the dataset better :](#getting-to-know-the-dataset-better-)
  - [Deciding on whether to use batches or not :](#deciding-on-whether-to-use-batches-or-not-)
- [Step 2 : Creating the neural network](#step-2--creating-the-neural-network)
  - [Deciding on Number of Hidden Layers and neurons :](#deciding-on-number-of-hidden-layers-and-neurons-)
  - [Creating the Neural network Sequence :](#creating-the-neural-network-sequence-)
- [Step 3 : Training the model on the dataset](#step-3--training-the-model-on-the-dataset)
- [Step 4 : Testing the Model](#step-4--testing-the-model)
- [Step 5 : Saving the model](#step-5--saving-the-model)
- [Step 6 : Logging of Parameters during Model Training and Testing](https://github.com/infinitecoder1729/mnist-dataset-classification/edit/main/README.md#step-6--logging-of-parameters-during-model-training-and-testing)
- [To View results for any random picture in the dataset, the following code can be used :](#to-view-results-for-any-random-picture-in-the-dataset-the-following-code-can-be-used-)
  - [Examples](#examples-)
  - [Model Accuracy](#model-accuracy--the-accuracy-of-the-model-with-this-code-is-approximately-978-to-9802-with-a-training-time-of-aprox-35-to-4-minutes)
- [Further Improvements](#further-improvements-)

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

### Creating the Neural network Sequence :

#### Definining the Model Sequence : 

Although a wide range of activation algorithms and formulations can be used and it can be discovered in depth. But for simplicity, LeakyReLU has been used for Hidden Layer [PyTorch LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html). The input layer and output have Linear activation [PyTorch Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html). Logsoftmax has been used to formulate the output [PyTorch LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)

The implementation is in : 

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/0fa674e4325acf4e82ea8513c948062677d04baf/MNIST%20Classification%20Model..py#L20-L23

#### Defining the loss function : 

Similar to above, many loss functions can be used to compute the loss but again for simplicity, NLLLoss i.e. Negatice Log Likelihood Loss has been used [PyTorch NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/0fa674e4325acf4e82ea8513c948062677d04baf/MNIST%20Classification%20Model..py#L25

## Step 3 : Training the model on the dataset

We have used SGD as Optimization Algorithm here with learning rate (lr) = 0.003 and momentum = 0.9 as suggested in general sense. [Typical lr values range from 0.0001 up to 1 and it is upon us to find a suitable value by cross validation

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/a014ffaeead36b9a8d1458b51b6f70fc3d8873e3/MNIST%20Classification%20Model..py#L33

To calculate the total training time, time module has been used. (Lines 34 and 48)

Trial and Error method can be used to find the suitable epoch value, for this code, it has been setup to be 18

Overall Training is being done as :

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/a014ffaeead36b9a8d1458b51b6f70fc3d8873e3/MNIST%20Classification%20Model..py#L33-L49

## Step 4 : Testing the Model 

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/a014ffaeead36b9a8d1458b51b6f70fc3d8873e3/MNIST%20Classification%20Model..py#L51-L66

## Step 5 : Saving the model 

https://github.com/infinitecoder1729/mnist-dataset-classification/blob/a014ffaeead36b9a8d1458b51b6f70fc3d8873e3/MNIST%20Classification%20Model..py#L68

## Step 6 : Logging of Parameters during Model Training and Testing

To log and vizualize the model parameters, Tensorboard has being used. For now, It logs Loss vs Epoch data for which graph can be accessed using :

```bash
tensorboard --logdir=runs
```

The Logging happens at :
https://github.com/infinitecoder1729/mnist-dataset-classification/blob/c4d559e6e3d4e49cbbaef084f0150a677c4e7408/MNIST%20Classification%20Model..py#L44

Following type of a graph is achieved as a result. It may vary if you change the algorithms and other parameters of the model :

![image](https://user-images.githubusercontent.com/77016507/236678310-bc09ca50-0e1f-4a05-84c6-8cb4b86d2142.png)


## To View results for any random picture in the dataset, the following code can be used :

It also creates a graph displaying the probabilities returned by the model.

```py
import numpy as np
def view_classify(img, ps):
    ps = ps.cpu().data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
img,label=train[np.random.randint(0,10001)] 
image=img.view(1, 784)
with tch.no_grad():
  logps = model(image)
ps = tch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(image.view(1, 28, 28), ps)
```

### Examples : 

![image](https://user-images.githubusercontent.com/77016507/225422901-908e96de-629f-4d33-b7ba-819960a97d66.png)

![image](https://user-images.githubusercontent.com/77016507/225423008-3f858a52-2331-48e1-b271-f6d6e25e2d91.png)

![image](https://user-images.githubusercontent.com/77016507/225423232-d0249b38-e191-495d-b9fd-8c32eb20da57.png)

### Model Accuracy : The Accuracy of the model with this code is approximately 97.8% to 98.02% with a training time of aprox. 3.5 to 4 minutes

## Further Improvements :

1. Working on expanding Logging and Graphing to Other Parameters to give a more comprehensive assessment of the model's performance.
2. Looking to test with different algorithms to strike a balance between training time and accuracy.

### Contributions, Suggestions, and inputs on logging and graphical representation for better understanding are welcome. 

# One of the trained model is uploaded to this repository as well for reference purposes.
