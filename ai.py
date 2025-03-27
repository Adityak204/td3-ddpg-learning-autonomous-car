import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import torch
import numpy as np
from torch.autograd import Variable
import random

# Global hyperparameters
BATCH_SIZE = 100


# Implementing Experience Replay without numpy dependencies
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, event):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Store as individual tensors
        self.memory[self.position] = (
            torch.FloatTensor(event[0]),
            torch.FloatTensor(event[1]),
            torch.FloatTensor(event[2]),
            torch.FloatTensor([event[3]]),
            torch.FloatTensor([event[4]]),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        # Return as separate tensors
        return (
            torch.stack([x[0] for x in samples]),
            torch.stack([x[1] for x in samples]),
            torch.stack([x[2] for x in samples]),
            torch.stack([x[3] for x in samples]),
            torch.stack([x[4] for x in samples]),
        )


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


# Implementing TD3 with pure PyTorch operations
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # TD3 parameters
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        self.discount = 0.99
        self.tau = 0.005

        self.total_it = 0
        self.replay_buffer = ReplayMemory(100000)
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.reward_window = []

    def select_action(self, state, noise=0.1):
        # Convert state to tensor without numpy
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().squeeze(0)

        if noise != 0:
            # Use PyTorch random instead of numpy
            noise_tensor = torch.randn_like(action) * noise
            action = action + noise_tensor
            action = torch.clamp(action, -self.max_action, self.max_action)

        return action.numpy().tolist()  # Convert to list to avoid numpy issues

    def train(self, batch_size=BATCH_SIZE):
        if len(self.replay_buffer.memory) < batch_size:
            return

        # Sample from replay buffer - already returns stacked tensors
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = (
            self.replay_buffer.sample(batch_size)
        )

        # Convert to proper device if using GPU
        state = batch_state
        next_state = batch_next_state
        action = batch_action
        reward = batch_reward
        done = batch_done

        # Rest of training code remains the same
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def update(self, reward, new_state):
        if self.last_state is not None:
            # Convert to numpy arrays first if they're tensors
            last_state = (
                self.last_state.numpy()
                if isinstance(self.last_state, torch.Tensor)
                else self.last_state
            )
            new_state = (
                new_state.numpy() if isinstance(new_state, torch.Tensor) else new_state
            )
            last_action = (
                self.last_action.numpy()
                if isinstance(self.last_action, torch.Tensor)
                else self.last_action
            )

            self.replay_buffer.push(
                (
                    last_state,
                    new_state,
                    last_action,
                    reward,
                    False,  # Assuming non-terminal state
                )
            )

        # Convert state to numpy array if it's a tensor
        new_state_np = (
            new_state.numpy() if isinstance(new_state, torch.Tensor) else new_state
        )
        action = self.select_action(np.array(new_state_np, dtype=np.float32))

        if len(self.replay_buffer.memory) > BATCH_SIZE:
            self.train()

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    def save(self, filename="td3_model.pth"):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename="td3_model.pth"):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
