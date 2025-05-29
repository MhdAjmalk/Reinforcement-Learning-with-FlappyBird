import pygame
import sys
import numpy as np
from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window

class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(800, 800)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    def reset(self):
        """Resets the game and returns the initial state"""
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        self.score_value = 0  # Add explicit score tracking
        self.player.set_mode(PlayerMode.NORMAL)
        return self.get_state()

    def step(self, action):
        """Enhanced step function with improved reward structure"""
        if action == 1:
            self.player.flap()

        # Store previous state
        prev_y = self.player.y
        prev_score = self.score_value
        
        # Get the next pipe
        next_upper_pipe = None
        next_lower_pipe = None
        for up_pipe, low_pipe in zip(self.pipes.upper, self.pipes.lower):
            if up_pipe.x + up_pipe.w > self.player.x:
                next_upper_pipe = up_pipe
                next_lower_pipe = low_pipe
                break
        
        if next_upper_pipe is None or next_lower_pipe is None:
            if self.pipes.upper and self.pipes.lower:
                next_upper_pipe = self.pipes.upper[0]
                next_lower_pipe = self.pipes.lower[0]
            else:
                return self.get_state(), -10.0, True  # Bigger penalty for no pipes
        
        # Calculate distances and positions
        gap_center_y = (next_lower_pipe.y + next_upper_pipe.y + next_upper_pipe.h) / 2
        prev_dist_to_gap = abs(prev_y - gap_center_y)

        # Update game state
        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        # Get new distances and check game state
        new_dist_to_gap = abs(self.player.y - gap_center_y)
        done = self.is_done()
        
        # Initialize base reward
        reward = 0.1  # Small positive reward for staying alive
        
        # Distance-based reward
        if new_dist_to_gap < prev_dist_to_gap:
            reward += 0.5  # Reward for moving towards gap
        else:
            reward -= 0.2  # Small penalty for moving away
            
        # Position-based rewards
        if new_dist_to_gap < 50:  # Close to gap center
            reward += 1.0
        elif new_dist_to_gap < 100:  # Reasonably close to gap
            reward += 0.5
            
        # Check if passed pipe
        if (self.player.x > next_upper_pipe.x + next_upper_pipe.w and 
            self.player.x <= next_upper_pipe.x + next_upper_pipe.w - next_upper_pipe.vel_x):
            self.score_value += 1
            reward += 10.0  # Big reward for passing pipe
            
        # Failure penalties
        if done:
            if self.player.crashed:
                reward = -10.0  # Bigger penalty for crash
            else:
                reward = -5.0  # Smaller penalty for other terminations
                
        return self.get_state(), reward, done

    def get_state(self):
        """Enhanced state representation with better normalization"""
        screen_height = self.config.window.height
        screen_width = self.config.window.width
        
        # Find next pipe
        next_upper_pipe = None
        next_lower_pipe = None
        for up_pipe, low_pipe in zip(self.pipes.upper, self.pipes.lower):
            if up_pipe.x + up_pipe.w > self.player.x:
                next_upper_pipe = up_pipe
                next_lower_pipe = low_pipe
                break
        
        if next_upper_pipe is None or next_lower_pipe is None:
            if self.pipes.upper and self.pipes.lower:
                next_upper_pipe = self.pipes.upper[0]
                next_lower_pipe = self.pipes.lower[0]
            else:
                return np.zeros(8, dtype=np.float32)
        
        # Calculate normalized features
        bird_y = (self.player.y - screen_height/2) / (screen_height/2)  # Center around 0
        bird_vel = self.player.vel_y / 10.0  # Normalize velocity
        
        # Pipe distances and positions
        pipe_dist_x = (next_upper_pipe.x - self.player.x) / screen_width
        gap_center_y = (next_lower_pipe.y + next_upper_pipe.y + next_upper_pipe.h) / 2
        gap_y_normalized = (gap_center_y - screen_height/2) / (screen_height/2)
        
        # Gap size and relative position
        gap_size = (next_lower_pipe.y - (next_upper_pipe.y + next_upper_pipe.h)) / screen_height
        dist_to_gap = (self.player.y - gap_center_y) / (screen_height/2)
        
        # Velocity features
        vel_towards_gap = -self.player.vel_y if self.player.y > gap_center_y else self.player.vel_y
        vel_normalized = vel_towards_gap / 15.0
        
        return np.array([
            bird_y,          # Normalized bird height relative to center
            bird_vel,        # Normalized bird velocity
            pipe_dist_x,     # Normalized horizontal distance to pipe
            gap_y_normalized,# Normalized gap vertical position
            dist_to_gap,     # Normalized distance to gap
            vel_normalized,  # Normalized velocity towards gap
            gap_size,       # Normalized gap size
            self.score_value / 10.0  # Normalized score
        ], dtype=np.float32)

    def is_done(self):
        """Enhanced game over detection with better termination conditions"""
        # Check collision
        if self.player.collided(self.pipes, self.floor):
            return True
            
        # Height boundaries
        if self.player.y < -2.5 * self.player.h:  # Too high
            return True
        if self.player.y > self.config.window.height - self.player.h:  # Too low
            return True
            
        # Check distance from pipe gap
        next_upper_pipe = None
        next_lower_pipe = None
        for up_pipe, low_pipe in zip(self.pipes.upper, self.pipes.lower):
            if up_pipe.x + up_pipe.w > self.player.x:
                next_upper_pipe = up_pipe
                next_lower_pipe = low_pipe
                break
                
        if next_upper_pipe and next_lower_pipe:
            gap_center = (next_lower_pipe.y + next_upper_pipe.y + next_upper_pipe.h) / 2
            if abs(self.player.y - gap_center) > 200:  # Increased tolerance
                return True
            
        return False

    def draw(self, screen):
        """Draw the game state to the screen"""
        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()
        pygame.display.update()
