import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.distributions import Categorical

import logging
import numpy as np
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp1_dims, mlp2_dims, mlp3_dims, mlp4_dims, lin1_dim, lin2_dim):
        super().__init__()
        self.input_dim = input_dim # input_dim is the variables in robot state + one human state
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims) # input is self.mlp1
        self.mlp3 = mlp(2*mlp1_dims[-1], mlp3_dims) # input is cat(self.mlp1, mean(self.mlp1))
        self.mlp4 = mlp(mlp2_dims[-1], mlp4_dims, last_relu=True) # input is sum(self.mlp3 * self.mlp2)
        self.lin1 = mlp(mlp4_dims[-1], [lin1_dim]) # input is self.mlp4
        self.lin2 = mlp(mlp4_dims[-1], [lin2_dim]) # input is self.mlp4
        
        # used when reformatting the mean of mlp1 output
        self.global_state_dim = mlp1_dims[-1]

    def forward(self, state, getPI=False):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
                      generated by MultiHumanRL.transform(state)
        :return:
        """
        size = state.shape # (batch_size, # humans, self.input_dim)
        # reshape input to be used in the network
        mlp1_output = self.mlp1(state.view((-1, size[2]))) # (batch_size * # humans, self.mlp1)
        mlp2_output = self.mlp2(mlp1_output) # (batch_size * # humans, self.mlp2)
        
        # compute mean across humans in each batch
        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True) # (batch_size, 1, self.mlp1)
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim) # (batch_size * # humans, self.mlp1)
        # calculate alphas
        mlp3_input = torch.cat([mlp1_output, global_state], dim=1) # (batch_size * # humans, 2 * self.mlp1)
        mlp3_output = self.mlp3(mlp3_input).view(size[0], size[1], 1).squeeze(dim=2) # (batch_size, # humans)
        alpha = softmax(mlp3_output, dim=1).unsqueeze(2) # (batch_size, # humans, 1)
        
        # calculate sum of mlp2 weighted by alpha and feed it through mlp4
        mlp4_input = torch.sum(torch.mul(alpha, mlp2_output.view(size[0], size[1], -1)), dim=1) # (batch_size, self.mlp2)
        mlp4_output = self.mlp4(mlp4_input) # (batch_size, self.mlp4)
        
        # calculate policy and value and return
        pi = softmax(self.lin1(mlp4_output), dim=1) # (batch_size, self.lin1)
        pi = Categorical(probs=pi)
        val = self.lin2(mlp4_output) # (batch_size, self.lin2)
        
        if getPI:
            return pi, val
        else:
            return val


class GAP(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'GAP'

    def configure(self, config):
        self.set_common_parameters(config)
        
        mlp1_dims = [int(x) for x in config.get('gap', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('gap', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('gap', 'mlp3_dims').split(', ')]
        mlp4_dims = [int(x) for x in config.get('gap', 'mlp4_dims').split(', ')]
        lin1_dim = config.getint('gap', 'lin1_dim')
        lin2_dim = config.getint('gap', 'lin2_dim')
        
        self.with_om = config.getboolean('gap', 'with_om')
        self.multiagent_training = config.getboolean('gap', 'multiagent_training')

        self.model = ValueNetwork(self.input_dim(), mlp1_dims, mlp2_dims, mlp3_dims, mlp4_dims, lin1_dim, lin2_dim) # uses MultiHumanRL.input_dim()

        logging.info('Policy: {}'.format(self.name))

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
            
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
            

        batch_states = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                          for human_state in state.human_states], dim=0)
        rotated_batch_input = self.rotate(batch_states).unsqueeze(0)

        pi, val = self.model(rotated_batch_input, getPI=True)
        a = pi.sample()
        log_pi = pi.log_prob(a).cpu().numpy()
        
        next_action = self.action_space[int(a.cpu().numpy())]
        
        if self.phase == 'train':
            self.last_state = self.transform(state)
        
        return next_action, pi, val, log_pi










