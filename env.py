import numpy as np
from src.flappy import Flappy
from datetime import datetime

# Current configuration
TIMESTAMP = "2025-05-28 13:36:28"
CURRENT_USER = "MhdAjmalk"

class FlappyBirdEnv:
    def __init__(self):
        # Initialize the game
        self.game = Flappy()
        
        # Track metrics
        self.total_steps = 0
        self.episode_steps = 0
        self.max_episode_steps = 3000
        self.previous_score = 0
        
        # Enhanced reward parameters
        self.alive_reward = 0.1
        self.death_penalty = -10.0
        self.pipe_reward = 15.0
        self.distance_reward_factor = 0.1
        self.consecutive_pipes_bonus = 5.0
        
        # State tracking
        self.pipes_cleared = 0
        self.last_distance = float('inf')
        self.best_score = 0

    def reset(self):
        """Reset the environment and return the initial state"""
        initial_state = self.game.reset()
        self.episode_steps = 0
        self.total_steps += 1
        self.previous_score = 0
        self.pipes_cleared = 0
        self.last_distance = float('inf')
        return self._normalize_state(initial_state)

    def _normalize_state(self, state):
        """Normalize state values for better learning"""
        try:
            state = np.array(state, dtype=np.float32)
            # Clip values to reasonable ranges
            state = np.clip(state, -1000, 1000)
            # Normalize to [-1, 1] range
            if np.any(state != 0):  # Avoid division by zero
                state = state / np.max(np.abs(state))
            return state
        except Exception as e:
            print(f"State normalization error: {e}")
            return state

    def _calculate_distance_reward(self):
        """Calculate reward based on distance to next pipe"""
        try:
            # Get bird's position and next pipe position
            bird_x = self.game.bird_x if hasattr(self.game, 'bird_x') else 0
            bird_y = self.game.bird_y if hasattr(self.game, 'bird_y') else 0
            pipe_x = self.game.next_pipe_x if hasattr(self.game, 'next_pipe_x') else 100
            pipe_y = self.game.next_pipe_y if hasattr(self.game, 'next_pipe_y') else 0
            
            # Calculate distance
            distance = np.sqrt((pipe_x - bird_x)**2 + (pipe_y - bird_y)**2)
            
            # Calculate reward based on distance improvement
            reward = (self.last_distance - distance) * self.distance_reward_factor
            self.last_distance = distance
            
            return reward
        except Exception:
            return 0

    # In env.py, update the step method:

    def step(self, action):
        """
        Execute action and return new state, reward, and done flag
        """
        self.episode_steps += 1
    
        # Get state from game
        next_state, base_reward, done = self.game.step(action)
        reward = 0.0  # Initialize as float
    
        # Handle game over
        if done:
            reward = float(self.death_penalty)  # Convert to float
        else:
            # Living reward
            reward = float(self.alive_reward)  # Convert to float
        
            # Score-based reward
            try:
               current_score = int(str(self.game.score))
               if current_score > self.previous_score:
                  reward += float(self.pipe_reward)  # Convert to float
                  self.pipes_cleared += 1
                  self.previous_score = current_score
            except (AttributeError, ValueError):
                pass
    
        # Clip reward to prevent numerical instability
        reward = np.clip(reward, -10.0, 10.0)
    
        # Check for episode termination
        if self.episode_steps >= self.max_episode_steps:
            done = True
    
        # Ensure state values are finite
        next_state = np.array(next_state, dtype=np.float32)
        next_state = np.clip(next_state, -100, 100)
    
        return next_state, reward, done
        
        
    def get_state(self):
        """Get the current normalized state"""
        state = self.game.get_state()
        return self._normalize_state(state)

    def render(self):
        """Render the current game state"""
        self.game.draw(self.game.config.screen)

    def is_done(self):
        """Check if the episode is complete"""
        return (self.game.is_done() or 
                self.episode_steps >= self.max_episode_steps)

    def get_info(self):
        """Get detailed environment information"""
        try:
            current_score = int(str(self.game.score))
        except (AttributeError, ValueError):
            current_score = 0
            
        return {
            'score': current_score,
            'best_score': self.best_score,
            'steps': self.episode_steps,
            'total_steps': self.total_steps,
            'pipes_cleared': self.pipes_cleared,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': CURRENT_USER,
            'episode_time': self.episode_steps / 30  # Assuming 30 FPS
        }
