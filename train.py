import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from datetime import datetime
import time
import pygame
import os
from env import FlappyBirdEnv
from agent import DQNAgent

# Configuration
TIMESTAMP = "2025-05-28 14:55:38"
CURRENT_USER = "MhdAjmalk"

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise KeyboardInterrupt("Game window closed")

def load_model(agent, model_path):
    """Safely load a model with error handling"""
    try:
        # Load with CPU if CUDA is not available
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Load state dictionaries
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get the score
        score = checkpoint.get('score', float('-inf'))
        print(f"Successfully loaded model from {model_path}")
        print(f"Model's best score: {score:.2f}")
        return score
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return float('-inf')

def train(render=True, game_speed=0.03, resume_training=True):
    print(f"Starting Flappy Bird Training at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: {CURRENT_USER}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    env = FlappyBirdEnv()
    agent = DQNAgent()
    
    best_score = float('-inf')
    
    # Try to load the previous best model
    if resume_training:
        try:
            # First try loading best_model_latest.pth
            if os.path.exists("best_model_latest.pth"):
                print("\nFound latest best model: best_model_latest.pth")
                best_score = load_model(agent, "best_model_latest.pth")
            else:
                # Look for score-based model files
                model_files = [f for f in os.listdir('.') if f.startswith('best_model_score_') and f.endswith('.pth')]
                if model_files:
                    # Sort by score
                    model_files.sort(key=lambda x: float(x.split('_')[3].replace('.pth', '')), reverse=True)
                    best_file = model_files[0]
                    print(f"\nFound previous best model: {best_file}")
                    best_score = load_model(agent, best_file)
                else:
                    print("\nNo previous model found. Starting fresh training.")
        except Exception as e:
            print(f"\nError while loading model: {str(e)}")
            print("Starting fresh training.")
            best_score = float('-inf')
    else:
        print("\nStarting fresh training (resume_training=False)")
    
    episodes = 500
    max_steps = 2000
    print_every = 1
    
    scores = []
    recent_scores = deque(maxlen=100)
    no_improvement = 0
    best_model_state = None
    
    try:
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0.0
            loss = 0.0
            
            print(f"\nStarting Episode {episode + 1}/{episodes}")
            print(f"Epsilon (exploration rate): {agent.epsilon:.4f}")
            
            for step in range(max_steps):
                if render:
                    handle_pygame_events()
                    env.render()
                    pygame.display.update()
                    time.sleep(game_speed)
                
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                if not np.isfinite(reward):
                    reward = 0.0
                
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    loss = agent.train_step()
                    if np.isnan(loss):
                        print("Warning: NaN loss detected, skipping update")
                        loss = 0.0
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            if np.isfinite(episode_reward):
                scores.append(episode_reward)
                recent_scores.append(episode_reward)
                avg_score = np.mean(recent_scores)
                
                if episode_reward > best_score:
                    best_score = episode_reward
                    no_improvement = 0
                    print(f"\n🌟 New Best Score: {best_score:.2f} 🌟")
                    
                    # Save best model state
                    best_model_state = {
                        'model_state_dict': agent.model.state_dict(),
                        'target_model_state_dict': agent.target_model.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'episode': episode,
                        'score': best_score,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'user': CURRENT_USER
                    }
                    
                    # Save with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    torch.save(best_model_state, f"best_model_score_{best_score:.2f}.pth")
                    # Also save as latest best model
                    torch.save(best_model_state, "best_model_latest.pth")
                else:
                    no_improvement += 1
            
            if episode % print_every == 0:
                print(f"\nEpisode: {episode + 1}/{episodes}")
                print(f"Score: {episode_reward:.2f}")
                print(f"Average Score: {avg_score:.2f}")
                print(f"Best Score: {best_score:.2f}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                print(f"Memory Size: {len(agent.memory)}")
                if loss != 0:
                    print(f"Loss: {loss:.6f}")
                print("-" * 50)
            
            if no_improvement >= 100 and episode > 100:
                print(f"\nStopping early: No improvement for 100 episodes")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Save final model state
        final_state = {
            'model_state_dict': agent.model.state_dict(),
            'target_model_state_dict': agent.target_model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode': episode,
            'best_score': best_score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': CURRENT_USER
        }
        torch.save(final_state, "final_model.pth")
        
        print("\nTraining Summary:")
        print(f"Episodes Completed: {episode + 1}")
        print(f"Best Score: {best_score:.2f}")
        if len(recent_scores) > 0:
            print(f"Final Average Score: {np.mean(recent_scores):.2f}")
        print(f"Final Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    try:
        train(render=True, game_speed=0.03, resume_training=True)
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
    finally:
        pygame.quit()
