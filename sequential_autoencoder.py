import numpy as np
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision import utils
import matplotlib.pyplot as plt

import MarioData as md
from torch.utils.data import DataLoader
from skimage import img_as_float

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim
import time
import torch.nn as nn
from tensorboardX import SummaryWriter

class Sequential_AutoEncoder(torch.nn.Module):

    def __init__(self):
        super(Sequential_AutoEncoder, self).__init__()
        # input is [1,84,84]
        # sequential/fully connected layers
        ## encoder ##
        #7056 ---> 128 
        self.fc1 = nn.Linear(84*84*1, 128)
        ## decoder ##
        # 128 ---> 7056
        self.fc2 = nn.Linear(128, 84*84*1)

    def forward(self, x):
        ## encode ##
        # hidden layer with relu activation function  
        x = F.relu(self.fc1(x))
        ## decode ##
        # second hidden layer/output with sigmoid
        x = F.sigmoid(self.fc2(x))
        return x

    @staticmethod
    def createLossAndOptimizer(net, learning_rate=0.001):
        #Loss function using MSE due to the task being reconstruction
        loss = torch.nn.MSELoss()
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        return(loss, optimizer)

        # 49152 ---> 128
        #   7056 for Grayscale 84,84, 21168
        #self.fc1 = nn.Linear(3*128*128, 128)
def trainNet(net, train_loader, val_loader, n_epochs, learning_rate, device, batchsize):
   
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    sum_path = 'src/summary/sequential'
    writer = SummaryWriter(sum_path)
    #Get training data
    n_batches = len(train_loader)
   
    #Create our loss and optimizer functions
    loss, optimizer = net.createLossAndOptimizer(net, learning_rate)
   
    #Time for printing
    training_start_time = time.time()
   
    #Loop for n_epochs
    for epoch in range(n_epochs):
       
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
       
        for i, data in enumerate(train_loader, 0):
            #Get inputs
            inputs = data['image'].to(device)
           
            #Wrap them in a Variable object
            inputs = Variable(inputs)

            #Set the parameter gradients to zero
            optimizer.zero_grad()
            #nk
            #https://stackoverflow.com/questions/49606482/how-to-resolve-runtime-error-due-to-size-mismatch-in-pytorch
            # input shape is (batch_size,3,128,128)
            # flattening to (batchsize, 49152) or (batchsize, 3*128*128)
            flattened_inputs = inputs.view(inputs.size(0),-1)
           
            outputs = net(flattened_inputs)

            # changing output shape back to input
            #outputs = outputs.view(batchsize, 3, 128, 128)
            outputs = outputs.view(batchsize, 1, 84, 84)


            loss_size = loss(outputs, inputs)
            loss_size.backward()
           
            optimizer.step()
           
            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()

           
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                # print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                #         epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                print(i)
               
                current_step_per = int(100 * (i+1) / n_batches)

                writer.add_scalar('AE_Train_Loss_{}'.format(epoch),running_loss / print_every, current_step_per)
               
                #writer.add_scalar('CAE_Train_Loss',running_loss / print_every, current_step_per)

                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% took: {:.2f}s'
                  .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), running_loss / print_every, 0, time.time() - start_time))
                #Reset running loss and time
               
                running_loss = 0.0
                start_time = time.time()
           
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(val_loader, 0):
            #Get inputs
            inputs = data['image'].to(device)
            #Wrap tensors in Variables
            inputs = Variable(inputs)
           
            # input shape is (batch_size,3,128,128)
            # flattening to (batchsize, 49152) or (batchsize, 3*128*128)
            flattened_inputs = inputs.view(inputs.size(0),-1)
            #input shape
           
            #Forward pass
           
            val_outputs = net(flattened_inputs)
            # changing output shape back to input
            val_outputs = val_outputs.view(batchsize, 1, 84, 84)
           
            val_loss_size = loss(val_outputs, inputs)
            total_val_loss += val_loss_size.data.item()
            current_step_per = int(100 * (i+1) / n_batches)
       
        writer.add_scalar('AE_Total_Train_Loss',total_train_loss,epoch+1)

        writer.add_scalar('AE_Val_Loss',total_val_loss, epoch+1)

        print("Validation loss = {:.2f}; Accuracy: {:.2f}%".format(total_val_loss / len(val_loader),0))
       
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def show_images(images):
    """Show image with Mario annotated for a batch of samples."""
    img = images
    grid = utils.make_grid(img)
    plt.imshow(img_as_float(grid.numpy().transpose((1, 2, 0))))        
    plt.axis('off')
    plt.ioff()
    plt.show()



def testNet(net, test_loader, device, batchsize):
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            #print(i)
            inputs = data['image'].to(device)
            inputs = Variable(inputs)
            # input shape is (batch_size,3,128,128)
            # flattening to (batchsize, 49152) or (batchsize, 3*128*128)
            flattened_inputs = inputs.view(inputs.size(0),-1)
            outputs = net(flattened_inputs)

            # changing output shape back to input
            outputs = outputs.view(batchsize, 1, 84, 84)
           
            final_inputs = inputs
            final_outputs = outputs            
           
        #show real images
        show_images(final_inputs)
        #show reconstructed images
        show_images(final_outputs)


if __name__ == "__main__":
    train_val_batch_size = 128
    test_batch_size =  4

    train_loader = DataLoader(dataset, batch_size=train_val_batch_size, sampler=train_sampler, num_workers=0, drop_last = True)
    validation_loader = DataLoader(dataset, batch_size=train_val_batch_size, sampler=valid_sampler, num_workers=0, drop_last = True)
    test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=0, drop_last = True)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    SAE = Sequential_AutoEncoder().to(device)
    print(SAE)
       
    trainNet(SAE, train_loader, validation_loader, n_epochs=10, learning_rate=0.001, device=device, batchsize = train_val_batch_size)

    testNet(SAE, test_loader, device=device, batchsize=test_batch_size)

    print("Done!")
