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

import os
import os.path
from os import path
import os
import argparse
import gym


from tensorboardX import SummaryWriter
from torchsummary import summary

class Convolutional_AutoEncoder(torch.nn.Module):
    # Input is (4, 84, 84)

    def __init__(self):
        super(Convolutional_AutoEncoder, self).__init__()
        ## encoder layers ##
        
        # conv layer (depth from 4 --> 16), 3x3 kernels
        
        
        self.conv1 = nn.Conv2d( 4, 16, 3, padding = 1)  
        # conv layer (depth from 16 --> 9), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 9, 3, padding = 1)
        
        ## decoder layers ##
        ## a kernel of 2 
        # no stride
        
        self.t_conv1 = nn.ConvTranspose2d(9, 16, 1)
        self.t_conv2 = nn.ConvTranspose2d(16,4, 1)


    def forward(self, x):
        
        ## encode ##
        # add hidden layers with relu activation function       
        x = F.relu(self.conv1(x))

        # add second hidden layer
        x = F.relu(self.conv2(x))
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))

        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
        #print(x.shape)

        return x

    @staticmethod
    def createLossAndOptimizer(net, learning_rate=0.001):
        #Loss function using MSE due to the task being reconstruction
        loss = torch.nn.MSELoss()
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        return(loss, optimizer)


def trainNet(net, train_loader, val_loader, n_epochs, learning_rate, device):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    sum_path = 'summary/CAE'
    writer = SummaryWriter(sum_path)
    #Get training data
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = net.createLossAndOptimizer(net, learning_rate)
    
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    
    
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    t_losses = []
    v_losses = []
    for epoch in range(n_epochs):
    
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        print('Length of train loader: {}'.format(len(train_loader)))
        for i, data in enumerate(train_loader, 0):
            #Get inputs
            inputs = data['image'].to(device)
            #print(i)
           
            #print('Data Image shape {}'.format(data['image'].shape))
            input_list = []
            for im in inputs:
                input_list.append(im) 
                #print(im.shape)
            stack_inputs = torch.stack(input_list)
            stacked_inputs = stack_inputs.view(1,4,84,84)
            #print('Stacked inputs: {}'.format(stacked_inputs.shape))
            #Wrap them in a Variable object
            #env, num_states, num_actions = create_train_env(args.world, args.stage)
            # [128,3,128,128]
            #data['image'] = cv2.cvtColor(data['image'], cv2.COLOR_RGB2GRAY)
            
            inputs = Variable(inputs)
            #inputs = inputs.reshape(1,12,128,128)
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            # output shape [128,3,128,128]
            outputs = net(stacked_inputs)
            #encoding pass for image 1
            #encoding for image 2
            # concat  output of encoder 1 and 2
            # update size of vector
            
            loss_size = loss(outputs, stacked_inputs)
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
                writer.add_scalar('CAE_Train_Loss',running_loss / print_every, current_step_per)
                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% took: {:.2f}s'
                  .format(epoch + 1, n_epochs, current_step_per, i + 1, len(train_loader), running_loss / print_every, 0, time.time() - start_time))
                #Reset running loss and time
                t_losses.append(running_loss / print_every)

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
            input_list = []
            for im in inputs:
                input_list.append(im) 
               # print(im.shape)
            stack_inputs = torch.stack(input_list)
            stacked_inputs = stack_inputs.view(1,4,84,84)
            
            inputs = Variable(inputs)
            #Forward pass
            val_outputs = net(stacked_inputs)
            val_loss_size = loss(val_outputs, stacked_inputs)
            total_val_loss += val_loss_size.data.item()
            current_step_per = int(100 * (i+1) / n_batches)
            if (i + 1) % (print_every + 1) == 0:
                
                writer.add_scalar('CAE_Val_Loss',val_loss_size.data.item(),current_step_per )
                v_losses.append(val_loss_size.data.item())

        #writer.add_scalar('CAE_Val_Loss',total_val_loss, epoch+1)
        
        print("Validation loss = {:.2f}; Accuracy: {:.2f}%".format(total_val_loss / len(val_loader),0))
    print('T Loss')
    print(t_losses)
    #0.028571737140370573, 0.004451130509685063, 0.0031637836057500134, 0.0021815441179376563, 0.001675358686333328, 0.0012807273321512002, 0.0011728005694436835, 0.001002008335534883, 0.0010308218211285349
    print('V Loss')
    print(v_losses)
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def show_images(images):
    """Show image with Mario annotated for a batch of samples."""
    img = images
    grid = utils.make_grid(img)
    plt.imshow(img_as_float(grid.numpy().transpose((1, 2, 0))))         
    plt.axis('off')
    plt.ioff()
    plt.show()



def testNet(net, test_loader, device):
    
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            inputs = data['image'].to(device)
            #inputs.reshape(4,3,128,128)
            #print(inputs.shape)
            #print(i)
            input_list = []
            
            
            for im in inputs:
                input_list.append(im)
                #print(im.shape)
                #print(im.shape)
            cat_inputs1 = torch.stack(input_list)
            cat_inputs2 = cat_inputs1.view(1,4,84,84)
            #print('CAT inputs: {}'.format(cat_inputs1.shape))

            #print('Stacked inputs: {}'.format(cat_inputs2.shape))

            inputs = Variable(inputs)
            
            outputs = net(cat_inputs2)
            #outputs=outputs.reshape(4,3,128,128) 
            outputs = outputs.view(4,1,84,84)
            final_inputs = cat_inputs2
            final_outputs = outputs            
            
        #show real images
        
        #show_images(cat_inputs1)
        show_images(cat_inputs1)

        #show reconstructed images
        show_images(final_outputs)


#if __name__ == "__main__":

def trainCAE(save_weights, save_model):
    os.environ['OMP_NUM_THREADS'] = '1'
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ********************************
    # Dataset stuff
    transform2apply = transforms.Compose([
                                            transforms.Resize((84,84)),
                                            transforms.Grayscale(),
                                            transforms.ToTensor()
                                        ])
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    dataset = md.DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    use_cuda = True

    # Creating data indices for training, validation and test splits:
    dataset_size = len(dataset)
    n_test = int(dataset_size * 0.05)
    n_train = dataset_size - 2 * n_test

    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:(n_train + n_test)]
    test_indices = indices[(n_train + n_test):]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    print("Train size: {}".format(len(train_sampler)))
    print("Validation size: {}".format(len(valid_sampler)))
    print("Test size: {}".format(len(test_sampler)))

    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=3, drop_last = True)
    validation_loader = DataLoader(dataset, batch_size=4, sampler=valid_sampler, num_workers=3, drop_last = True)
    test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler, num_workers=3, drop_last = True)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    
    print('initializing model')
    CAE_model = Convolutional_AutoEncoder().to(device)  
    print("Model's state_dict:")
    for param_tensor in CAE_model.state_dict():
        print(param_tensor, "\t", CAE_model.state_dict()[param_tensor].size())
    '''
    if not path.exists(save_weights) and not path.exists(save_model):
        trainNet(CAE_model, train_loader, validation_loader, n_epochs=2, learning_rate=0.001, device=device)
        #save training model    
        torch.save(CAE_model.state_dict(),save_weights), 
        torch.save(CAE_model, save_model)
    else:
        #load trained model weights
        CAE_model.load_state_dict(torch.load(save_weights), strict = False)
        CAE_model.eval()
    '''
    #print(summary(CAE_model, input_size=(4,84,84), batch_size=4, device='cpu'))
    print('training model')
    print(CAE_model)
    trainNet(CAE_model, train_loader, validation_loader, n_epochs=1, learning_rate=0.001, device=device)
    
    testNet(CAE_model, test_loader, device=device)

    print("Done!")


if __name__ == "__main__":
    
    # check if pretrained auto_encoder directory exists
    trained_cae_dir = 'pretrained/trained_cae'
    trained_cae_weights = '/trained_cae_weights12-gray.pth'
    trained_cae_full_model = '/trained_cae_model12-gray.pth'

    if not os.path.isdir(trained_cae_dir):
        os.makedirs(trained_cae_dir)
    # train convolutional auto encoder
    # if pretrained weights exist test reconstruction    
    trainCAE((trained_cae_dir+trained_cae_weights), (trained_cae_dir+trained_cae_full_model))
