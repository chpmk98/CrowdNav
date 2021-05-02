import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from labml_nn.rl.ppo import ClippedPPOLoss, ClippedValueFunctionLoss


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.ppo_loss = ClippedPPOLoss()
        self.value_loss = ClippedValueFunctionLoss()

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def set_optimizer(self, optimizer, learning_rate, epsilon=None):
        logging.info('Current learning rate: %f', learning_rate)
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=epsilon)
        else:
            raise NotImplementedError

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                doPPO = len(data) > 2
                if doPPO:
                    inputs, aas, actions, values, log_pis, advantages = data
                    aas = Variable(aas)
                    log_pis = Variable(log_pis)
                    advantages = Variable(advantages)
                else:
                    inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                if doPPO:
                    # loss for PPO
                    sampled_return = values + advantages
                    sampled_normalized_advantage = (advantages - advantages.mean())/(advantages.std() + 1e-8)
                    pi, val = self.model(inputs, getPI=True)
                    log_pi = pi.log_prob(aas)
                    # calculate policy loss
                    policy_loss = self.ppo_loss(log_pi, log_pis, sampled_normalized_advantage, 0.1)
                    # calculate Entropy Bonus
                    entropy_bonus = pi.entropy()
                    entropy_bonus = entropy_bonus.mean()
                    # calculate value function loss
                    value_loss = self.value_loss(val, values, sampled_return, 0.1)
                    # calculate total loss
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, values)
                loss.backward()
                # clip gradients
                if doPPO:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            data = next(iter(self.data_loader))
            doPPO = len(data) > 2
            if doPPO:
                inputs, actions, values, log_pis, advantages = data
                actions = Variable(actions)
                log_pis = Variable(log_pis)
                advantages = Variable(advantages)
            else:
                inputs, values = data
            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()
            if doPPO:
                # loss for PPO
                sampled_return = values + advantages
                sampled_normalized_advantage = (advantages - advantages.mean())/(advantages.std() + 1e-8)
                _, pi, val, _ = self.model(inputs)
                log_pi = pi.log_prob(actions)
                # calculate policy loss
                policy_loss = self.ppo_loss(log_pi, log_pis, sampled_normalized_advantage, 0.1)
                # calculate Entropy Bonus
                entropy_bonus = pi.entropy()
                entropy_bonus = entropy_bonus.mean()
                # calculate value function loss
                value_loss = self.value_loss(val, values, sampled_return, 0.1)
                # calculate total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
            loss.backward()
            # clip gradients
            if doPPO:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
