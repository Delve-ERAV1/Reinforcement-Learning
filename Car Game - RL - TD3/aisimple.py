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

p = 64

writer = SummaryWriter()

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
    batch_states, batch_next_states, batch_actions, batch_rewards = [], [], [], []
    for i in ind:
      state, next_state, action, reward = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))

    batch_states = np.array(batch_states)
    batch_next_states = np.array(batch_next_states)
    batch_actions = np.array(batch_actions)
    batch_rewards = np.array(batch_rewards).reshape(-1, 1)

    return batch_states, batch_next_states, batch_actions, batch_rewards

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, p)
    self.layer_2 = nn.Linear(p, p*8)
    self.layer_3 = nn.Linear(p*8, p*8)
    self.layer_4 = nn.Linear(p*8, action_dim)

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = F.relu(self.layer_3(x))
    return  5 * torch.tanh(self.layer_4(x))
  

class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, p)
    self.layer_2 = nn.Linear(p, p*8)
    self.layer_3 = nn.Linear(p*8, p*8)
    self.layer_4 = nn.Linear(p*8, action_dim)
    # Defining the second Critic neural network
    self.layer_11 = nn.Linear(state_dim + action_dim, p)
    self.layer_12 = nn.Linear(p, p*8)
    self.layer_13 = nn.Linear(p*8, p*8)
    self.layer_14 = nn.Linear(p*8, action_dim)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = F.relu(self.layer_3(x1))
    x1 = self.layer_4(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_11(xu))
    x2 = F.relu(self.layer_12(x2))
    x2 = F.relu(self.layer_13(x2))
    x2 = self.layer_14(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = F.relu(self.layer_3(x1))
    x1 = self.layer_4(x1)
    return x1

# Building the whole Training Process into a class

 # Random seed number
 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
 # How often the evaluation step is performed (after how many timesteps)
 # Total number of iterations/timesteps
 # Boolean checker whether or not to save the pre-trained model


class TD3(object):

  def __init__(self, state_dim, action_dim, max_action, batch_size=200, discount=0.99, 
               tau=0.005, policy_noise=0.2, noise_clip=0.5, expl_noise=0.1, 
               save_models=True, seed=0):
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
       "last_state": [round(random.uniform(0, 1), 2) for _ in range(state_dim)], 
       "last_action": [round(random.uniform(*self.max_action[0]), 2), round(random.uniform(*self.max_action[1]), 2)],
       "last_reward": -1
    }

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device) #todo explore action space
    action = self.actor(state).squeeze(0).detach()
    return action

  def train(self):

    # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
    batch_states, batch_next_states, batch_actions, batch_rewards = self.replay_buffer.sample(self.batch_size)
    state = torch.Tensor(batch_states).to(device)
    next_state = torch.Tensor(batch_next_states).to(device)
    action = torch.Tensor(batch_actions).to(device)
    reward = torch.Tensor(batch_rewards).to(device)

    # Step 5: From the next state s’, the Actor target plays the next action a’
    next_action = self.actor_target(next_state)

    # Define noise parameters
    policy_noise = self.params["policy_noise"]
    noise_clip = self.params["noise_clip"]

    # Generate Gaussian noise for each action component
    noise = torch.randn_like(next_action) * policy_noise
    noise = noise.clamp(-noise_clip, noise_clip)

    # Apply noise to the next action and clamp within the appropriate range
    next_action = (next_action + noise).clamp(-5, 5)

    # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
    target_Q1, target_Q2 = self.critic_target(next_state, next_action)

    # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
    target_Q = torch.min(target_Q1, target_Q2)

    # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
    target_Q = reward + (self.params["discount"] * target_Q).detach()

    # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
    current_Q1, current_Q2 = self.critic(state, action.unsqueeze(1))

    # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    writer.add_scalar("critic_loss", critic_loss, self.timesteps['total_timesteps'])

    # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
    if self.timesteps['total_timesteps'] % self.freq['policy_freq'] == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
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


  def update(self, reward, new_signal):

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
        action = round(random.uniform(*self.max_action[0]), 2)
    else: # After n timesteps, we switch to the model
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        action = self.select_action(new_state)#.squeeze(0).detach().numpy()

        # If the explore_noise parameter is not 0, we add noise to the action and we clip it
        if self.params["expl_noise"] != 0:
            action = (action[0].cpu().numpy() + np.random.normal(0, self.params["expl_noise"], size=1)).clip(-5, 5).item()
        else:
           action = action.item()

    # We store the new transition into the Experience Replay memory (ReplayBuffer)
    if self.timesteps['timesteps_since_eval'] != 0:
        self.replay_buffer.add((self.last_values['last_state'], new_signal, self.last_values['last_action'], self.last_values['last_reward']))

    self.last_values['last_state'] = new_signal
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