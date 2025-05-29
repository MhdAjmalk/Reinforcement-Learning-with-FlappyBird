import torch
import os
import numpy as np
from agent import Agent
from env import FlappyBirdEnv
import time
import glob
from datetime import datetime

# Current configuration
TIMESTAMP = "2025-05-28 10:10:36"
CURRENT_USER = "MhdAjmalk"

def get_latest_model():
    """Get the path to the latest model"""
    model_dirs = glob.glob("models_*")
    if not model_dirs:
        raise FileNotFoundError("No model directories found!")
    
    # Sort directories by creation time to get the most recent
    latest_dir = max(model_dirs, key=os.path.getctime)
    print(f"Found latest model directory: {latest_dir}")
    
    # Try best model first, then final model, then checkpoints
    model_paths = [
        os.path.join(latest_dir, "best_model.pth"),
        os.path.join(latest_dir, "final_model.pth")
    ]
    model_paths.extend(sorted(glob.glob(os.path.join(latest_dir, "checkpoint_*.pth"))))
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Using model: {path}")
            return path
    
    raise FileNotFoundError(f"No model files found in {latest_dir}")

def evaluate_single_episode(model_path, render=True, delay=0.05):
    """Evaluate the model for a single episode"""
    env = FlappyBirdEnv()
    agent = Agent()
    
    try:
        # Load model with proper error handling
        state_dict = torch.load(model_path, map_location=agent.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            agent.model.load_state_dict(state_dict['model_state_dict'])
        else:
            agent.model.load_state_dict(state_dict)
        
        agent.model.eval()  # Important: set to evaluation mode
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return 0
    
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("Starting evaluation...")
    
    try:
        while not done:
            if render:
                env.render()
                if delay > 0:
                    time.sleep(delay)
            
            # Get action (always exploit in evaluation)
            with torch.no_grad():
                action = agent.get_action(state, exploit_only=True)
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return total_reward
    
    return total_reward

def evaluate_multiple_episodes(model_path, num_episodes=5, render=True, delay=0.05):
    """Evaluate the model across multiple episodes"""
    print(f"\nEvaluating model across {num_episodes} episodes...")
    print("=" * 50)
    
    scores = []
    best_score = float('-inf')
    best_episode = 0
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            score = evaluate_single_episode(model_path, render=render, delay=delay)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_episode = episode + 1
            
            print(f"Episode {episode + 1} Score: {score}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        if not scores:
            return []
    
    # Calculate and display statistics
    avg_score = np.mean(scores)
    worst_score = min(scores)
    std_score = np.std(scores)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {best_score} (Episode {best_episode})")
    print(f"Worst Score: {worst_score}")
    print(f"Standard Deviation: {std_score:.2f}")
    print(f"All Scores: {scores}")
    
    # Save results
    results_dir = os.path.dirname(model_path)
    results_file = os.path.join(results_dir, "evaluation_results.txt")
    
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write(f"Best Score: {best_score} (Episode {best_episode})\n")
        f.write(f"Worst Score: {worst_score}\n")
        f.write(f"Standard Deviation: {std_score:.2f}\n")
        f.write(f"All Scores: {scores}\n")
        f.write(f"Evaluated by: {CURRENT_USER}\n")
    
    return scores

def evaluate(model_path=None, num_episodes=10, render=True, delay=0.05):
    """Main evaluation function"""
    print(f"\nFlappy Bird AI Evaluation")
    print(f"Current Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluator: {CURRENT_USER}")
    
    if model_path is None:
        try:
            model_path = get_latest_model()
            print(f"Using latest model: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []
    
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Delay: {delay}s")
    print("-" * 30)
    
    return evaluate_multiple_episodes(
        model_path=model_path,
        num_episodes=num_episodes,
        render=render,
        delay=delay
    )

if __name__ == "__main__":
    try:
        # Find latest model
        latest_model = get_latest_model()
        print(f"Latest model found: {latest_model}")
        
        # Run evaluation
        scores = evaluate(
            model_path=latest_model,
            num_episodes=10,
            render=True,
            delay=0.05
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have trained models in the models_* directories.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
