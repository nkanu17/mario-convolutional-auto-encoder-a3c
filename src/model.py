
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import math


class Convolutional_AutoEncoder(torch.nn.Module):
    # Input is (3, 128, 128)

    def __init__(self):
        super(Convolutional_AutoEncoder, self).__init__()
        ## encoder layers ##
        
        # conv layer (depth from 3 --> 16), 3x3 kernels
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
        #x = F.relu(self.t_conv1(x))

        # output layer (with sigmoid for scaling from 0 to 1)
        #x = F.sigmoid(self.t_conv2(x))


        return x

    @staticmethod
    def createLossAndOptimizer(net, learning_rate=0.001):
        #Loss function using MSE due to the task being reconstruction
        loss = torch.nn.MSELoss()
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        return(loss, optimizer)




class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # num_inputs = num of states
        #print('INPUTS')
        #print(num_inputs)
        # input shape = [1,4,84,84]
        # input [1,12,128,128]
        # the 9 input channels are coming from encoder
        self.conv1 = nn.Conv2d(9, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        #2048 not 1152 28224
        #self.lstm = nn.LSTMCell(28224, 512)
        # output from conv2 is 14112 with just 
        self.lstm = nn.LSTMCell(14112, 512)
        #self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        
        '''
        print('Forward x------------------')
        print('x shape {}'.format(x.shape))
        
        print('hx shape {}'.format(hx.shape))
        print('cx shape {}'.format(cx.shape))
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.t_conv1(x))
        #x = F.sigmoid(self.t_conv2(x))
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        '''
        print('output x--------------------')
        print(x.shape)
        print(hx.shape)
        print(cx.shape)
        '''
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx



class A3C_original(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(A3C_original, self).__init__()
        
        # the 9 input channels are coming from encoder
        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        #2048 not 1152 28224
        #self.lstm = nn.LSTMCell(28224, 512)
        # output from conv2 is 14112
        self.lstm = nn.LSTMCell(1152, 512)
        #self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        
        '''
        print('Forward x------------------')
        print('x shape {}'.format(x.shape))
        
        print('hx shape {}'.format(hx.shape))
        print('cx shape {}'.format(cx.shape))
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        '''
        print('output x--------------------')
        print(x.shape)
        print(hx.shape)
        print(cx.shape)
        '''
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx




    
class SharedAdam(optim.Adam):
    def __init__(self, params, lr):
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
