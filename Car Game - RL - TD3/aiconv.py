# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter




writer = SummaryWriter()


fc_size, down_x, neurons = 128, 32//4, 4

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)
      self.ptr = (self.ptr + 1) % self.max_size

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_grid, batch_next_states, batch_next_grid, batch_actions, batch_rewards = [], [], [], [], [], []
    for i in ind:
      state, next_state, action, reward = self.storage[i]

      batch_states.append(np.array(state[0], copy=False))
      batch_grid.append(state[1])

      batch_next_states.append(np.array(next_state[0], copy=False))
      batch_next_grid.append(next_state[1])

      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))

    batch_states = [np.array(batch_states), batch_grid]
    batch_next_states = [np.array(batch_next_states), batch_next_grid]
    
    batch_actions = np.array(batch_actions)
    batch_rewards = np.array(batch_rewards).reshape(-1, 1)

    return batch_states, batch_next_states, batch_actions, batch_rewards


class Actor(nn.Module):
    def __init__(self, num_additional_features, action_dim):
        super(Actor, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, neurons, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(neurons, neurons*2, kernel_size=3, stride=2, padding=1)
        # Flattening
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Linear(neurons*2 * down_x * down_x + num_additional_features, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size//2)
        self.fc3 = nn.Linear(fc_size//2, action_dim)

    def forward(self, spatial_input, additional_features):
        # Process spatial input
        x = F.relu(self.conv1(spatial_input))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        # Combine with additional features
        combined = torch.cat([x, additional_features], dim=1)
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action

class Critic(nn.Module):
    def __init__(self, num_additional_features, action_dim):
        super(Critic, self).__init__()
        # Convolutional layers 1st Critic
        self.conv11 = nn.Conv2d(1, neurons, kernel_size=3, stride=2, padding=1)
        self.conv21 = nn.Conv2d(neurons, neurons*2, kernel_size=3, stride=2, padding=1)
        # Flattening
        self.flatten1 = nn.Flatten()
        # Fully connected layers
        self.fc11 = nn.Linear(neurons*2 * down_x * down_x + num_additional_features + action_dim, fc_size)
        self.fc12 = nn.Linear(fc_size, fc_size//2)
        self.fc13 = nn.Linear(fc_size//2, 1)

        # Convolutional layers 2nd Critic
        self.conv12 = nn.Conv2d(1, neurons, kernel_size=3, stride=2, padding=1)
        self.conv22 = nn.Conv2d(neurons, neurons*2, kernel_size=3, stride=2, padding=1)
        # Flattening
        self.flatten2 = nn.Flatten()
        # Fully connected layers
        self.fc21 = nn.Linear(neurons*2 * down_x * down_x + num_additional_features + action_dim, fc_size)
        self.fc22 = nn.Linear(fc_size, fc_size//2)
        self.fc23 = nn.Linear(fc_size//2, 1)

    def forward(self, spatial_input, additional_features, action):
        # Forward-Propagation on the first Critic Neural Network
        # Process spatial input
        x1 = F.relu(self.conv11(spatial_input))
        x1 = F.relu(self.conv21(x1))
        x1 = self.flatten1(x1)
        # Combine with additional features and action
        combined1 = torch.cat([x1, additional_features, action], dim=1)
        # Fully connected layers
        x1 = F.relu(self.fc11(combined1))
        x1 = F.relu(self.fc12(x1))
        value1 = self.fc13(x1)

        # Forward-Propagation on the Second Critic Neural Network
        # Process spatial input
        x2 = F.relu(self.conv12(spatial_input))
        x2 = F.relu(self.conv22(x2))
        x2 = self.flatten2(x2)
        # Combine with additional features and action
        combined2 = torch.cat([x2, additional_features, action], dim=1)
        # Fully connected layers
        x2 = F.relu(self.fc21(combined2))
        x2 = F.relu(self.fc22(x2))
        value2 = self.fc23(x2)
        return value1, value2
    
    def Q1(self, spatial_input, additional_features, action):
        x1 = F.relu(self.conv11(spatial_input))
        x1 = F.relu(self.conv21(x1))
        x1 = self.flatten1(x1)
        # Combine with additional features and action
        combined1 = torch.cat([x1, additional_features, action], dim=1)
        # Fully connected layers
        x1 = F.relu(self.fc11(combined1))
        x1 = F.relu(self.fc12(x1))
        value1 = self.fc13(x1)
        return(value1)


# Building the whole Training Process into a class

 # Random seed number
 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
 # How often the evaluation step is performed (after how many timesteps)
 # Total number of iterations/timesteps
 # Boolean checker whether or not to save the pre-trained model


class TD3(object):

  def __init__(self, state_dim, action_dim, max_action, batch_size=100, discount=0.9, 
               tau=0.005, policy_noise=0.2, noise_clip=0.5, expl_noise=0.5, 
               save_models=False, seed=0):
    self.actor = Actor(state_dim, action_dim).to(device)
    self.actor_target = Actor(state_dim, action_dim).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.replay_buffer = ReplayBuffer()
    self.max_action = max_action # [(-5, 5), (0.5, 2)]
    self.batch_size = batch_size
    #self.evaluations = [self.evaluate_policy()]

    self.timesteps = {k[0]:k[1] for k in [('total_timesteps', 0), ('timesteps_since_eval', 0), 
                                          ('start_timesteps', 1e4), ('max_timesteps', 0)]}
    
    self.freq = {'policy_freq': 2, 'eval_freq': 5e3}

    self.params = {
       "policy_noise": policy_noise, "discount": discount,
       "noise_clip": noise_clip, "tau": tau, "expl_noise": expl_noise,
    }
    
    self.save_models=save_models
    self.seed=seed
    self.file_name = "TD3-torch.pth"

    self.last_action = 0
    self.last_reward = 0
    self.reward_window = []

    self.last_values = {
       "last_state": [[round(random.uniform(0, 1), 2) for _ in range(state_dim)], np.ones((32, 32))], 
       "last_action": [round(random.uniform(*self.max_action[0]), 2), round(random.uniform(*self.max_action[1]), 2)],
       "last_reward": -1
    }


  def select_action(self, state):
    grid, state = state
    state = torch.Tensor(state.reshape(1, -1)).to(device) #todo explore action space
    grid = torch.tensor(grid).to(device)

    action = self.actor(grid, state).squeeze(0).detach()
    # return [self.max_action[0][1] * torch.tanh(action[0]), 
    #         self.max_action[1][1] * torch.sigmoid(action[1])]
    return action

  def train(self):

    # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
    batch_states, batch_next_states, batch_actions, batch_rewards = self.replay_buffer.sample(self.batch_size)
    state = torch.Tensor(batch_states[0]).to(device)
    grid = torch.tensor(np.array(batch_states[1])).to(device)

    next_state = torch.Tensor(batch_next_states[0]).to(device)
    next_grid = torch.tensor(np.array(batch_next_states[1])).to(device)

    action = torch.Tensor(batch_actions).to(device)
    reward = torch.Tensor(batch_rewards).to(device)

    # Step 5: From the next state s’, the Actor target plays the next action a’
    next_action = self.actor_target(next_grid, next_state)

    # Define noise parameters
    policy_noise = self.params["policy_noise"]
    noise_clip = self.params["noise_clip"]

    # Generate Gaussian noise for each action component
    noise = torch.randn_like(next_action) * policy_noise
    noise = noise.clamp(-noise_clip, noise_clip)

    # Apply noise to the next action and clamp within the appropriate range
    next_action = (next_action + noise).clamp(-5, 5)
    # next_action = torch.cat([
    #     next_action[:, 0].clamp(self.max_action[0][0], self.max_action[0][1]).unsqueeze(1),
    #     next_action[:, 1].clamp(self.max_action[1][0], self.max_action[1][1]).unsqueeze(1)
    # ], dim=1)
    #next_action = next_action.clamp(-5, 5)#.unsqueeze(1)
    # def forward(self, spatial_input, additional_features, action):

    # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
    target_Q1, target_Q2 = self.critic_target(next_grid, next_state, next_action)

    # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
    target_Q = torch.min(target_Q1, target_Q2)

    # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
    target_Q = reward + (self.params["discount"] * target_Q).detach()

    # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
    current_Q1, current_Q2 = self.critic(grid, state, action.unsqueeze(1))

    # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    writer.add_scalar("critic_loss", critic_loss, self.timesteps['total_timesteps'])

    # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
    if self.timesteps['total_timesteps'] % self.freq['policy_freq'] == 0:
        actor_loss = -self.critic.Q1(grid, state, self.actor(grid, state)).mean()
        writer.add_scalar("actor_loss", actor_loss, self.timesteps['total_timesteps'])
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

    writer.flush()

  def update(self, reward, new_signal):

    grid_info = new_signal[-1]
    new_signal = new_signal[:-1]

    if self.timesteps['total_timesteps'] > 400:
        print("Total Timesteps: {} Since Eval: {} Reward: {} Total Sum {}".format(self.timesteps['total_timesteps'], 
                                                                      self.timesteps['timesteps_since_eval'], reward, 
                                                                      sum(self.reward_window)/len(self.reward_window)+1
                                                                      ))
        self.train()

    # We evaluate and we save the policy uniform
    if self.timesteps['timesteps_since_eval'] >= self.freq['eval_freq']:
        self.timesteps['timesteps_since_eval'] %= self.freq['eval_freq']
        #self.evaluations.append(self.evaluate_policy())
        self.save(self.file_name, directory="./pytorch_models")
        #np.save("./results/%s" % (self.file_name), self.evaluations)

    # Before 10000 timesteps, we play random actions
    if self.timesteps['total_timesteps'] < self.timesteps['start_timesteps']:
        # action = [round(random.uniform(*self.max_action[0]), 2), 
        #           round(random.uniform(*self.max_action[1]), 2)]
        action = round(random.uniform(*self.max_action[0]), 2)
    else: # After n timesteps, we switch to the model
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        new_grid = torch.tensor(grid_info).unsqueeze(0)
        action = self.select_action([new_grid, new_state])#.squeeze(0).detach().numpy()

        # If the explore_noise parameter is not 0, we add noise to the action and we clip it
        if self.params["expl_noise"] != 0:
            #steering_action = (action[0].numpy() + np.random.normal(0, self.params["expl_noise"], size=1)).clip(-5, 5)
            #accel_action = (action[1].numpy() + np.random.normal(0, self.params["expl_noise"], size=1)).clip(0.5, 2.0)
            #action = [steering_action.item(), accel_action.item()]
            action = (action[0].cpu().numpy() + np.random.normal(0, self.params["expl_noise"], size=1)).clip(-5, 5).item()
        else:
           #action = [action[0].item(), action[1].item()]
           action = action.item()

    # We store the new transition into the Experience Replay memory (ReplayBuffer)
    if self.timesteps['timesteps_since_eval'] != 0:
        self.replay_buffer.add((self.last_values['last_state'], [new_signal, grid_info], 
                                self.last_values['last_action'], self.last_values['last_reward']))

    self.last_values['last_state'] = [new_signal, grid_info]
    self.last_values['last_reward'] = reward
    self.last_values['last_action'] = action

    self.reward_window.append(reward)

    self.timesteps['total_timesteps'] += 1
    self.timesteps['timesteps_since_eval'] += 1
    return(action)
  
  def score(self):
    return sum(self.reward_window)/(len(self.reward_window)+1.)


    # def evaluate_policy(self, state, eval_episodes=10):
    #     avg_reward = 0.
    #     for _ in range(eval_episodes):
    #         action = self.select_action(np.array(state))
    #         obs, reward, done, _ = env.step(action)
    #         avg_reward += reward
    #     avg_reward /= eval_episodes
    #     print ("---------------------------------------")
    #     print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    #     print ("---------------------------------------")
    #     return avg_reward


  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))