import torch as tch
import torchvision.datasets as dt
import torchvision.transforms as trans
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time

train = dt.MNIST(root="./datasets", train=True, transform=trans.ToTensor(), download=True)
test = dt.MNIST(root="./datasets", train=False, transform=trans.ToTensor(), download=True)
print("No. of Training examples: ",len(train))
print("No. of Test examples: ",len(test))

train_batch = tch.utils.data.DataLoader(train, batch_size=30, shuffle=True)

input = 784
hidden = 490
output = 10

model = nn.Sequential(nn.Linear(input, hidden),
                      nn.LeakyReLU(),
                      nn.Linear(hidden, output),
                      nn.LogSoftmax(dim=1))

lossfn = nn.NLLLoss()
images, labels = next(iter(train_batch))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = lossfn(logps, labels)
loss.backward()

optimize = tch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time_start = time()
epochs = 18
for num in range(epochs):
    run=0
    for images, labels in train_batch:
        images = images.view(images.shape[0], -1)
        optimize.zero_grad()
        output = model(images)
        loss = lossfn(output, labels)
        writer.add_scalar("Loss", loss, num)
        loss.backward()
        optimize.step()
        run += loss.item()
    else:
        print("Epoch Number : {} = Loss : {}".format(num, run/len(train_batch)))
Elapsed=(time()-time_start)/60
print("\nTraining Time (in minutes) : ",Elapsed)
writer.flush()
writer.close()

correct=0
all = 0
for images,labels in test:
  img = images.view(1, 784)
  with tch.no_grad():
    logps = model(img)   
  ps = tch.exp(logps)
  probab = list(ps.numpy()[0])
  prediction = probab.index(max(probab))
  truth = labels
  if(truth == prediction):
    correct += 1
  all += 1

print("Number Of Images Tested : ", all)
print("Model Accuracy : ", (correct/all))

tch.save(model, './mnist_model.pt')
