import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

# Configuration
TIMESTAMP = "2025-05-28 13:52:26"
CURRENT_USER = "MhdAjmalk"

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        
        # Enhanced network architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:  # Changed from Agent to DQNAgent
    def __init__(self, state_size=8, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Network parameters
        self.hidden_size = 128
        
        # Create main network and target network
        self.model = DQN(state_size, self.hidden_size, action_size).to(self.device)
        self.target_model = DQN(state_size, self.hidden_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Learning parameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.target_update = 10
        
        # Experience replay
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.min_memory_size = 1000
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training stats
        self.n_games = 0
        self.training_step = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, exploit_only=False):
        """Get action using epsilon-greedy policy"""
        # Convert state to tensor
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if not exploit_only and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Get action from model
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        
        return torch.argmax(action_values).item()

    def train_step(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.min_memory_size:
            return 0.0
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and ensure finite values
        states = torch.FloatTensor(np.clip(states, -100, 100)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(np.clip(rewards, -10, 10)).to(self.device)
        next_states = torch.FloatTensor(np.clip(next_states, -100, 100)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_Q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_Q = self.target_model(next_states).max(1)[0]
            target_Q = rewards + (1 - dones) * self.gamma * next_Q
        
        # Clip target Q values
        target_Q = torch.clamp(target_Q, -100, 100)
        
        # Compute loss
        loss = nn.MSELoss()(current_Q.squeeze(), target_Q)
        
        # Check for NaN loss
        if torch.isnan(loss):
            return 0.0
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
