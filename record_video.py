import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import argparse
import sys

# Import the agent classes from the main module
from ppo_cartpole import PPOAgent, ActorCritic


def record_video_demo(model_path, epsilon=0.2, num_episodes=5, video_dir='./results/videos', max_steps_per_episode=1000):
    os.makedirs(video_dir, exist_ok=True)
    
    # Create environment
    print(f"Creating CartPole-v1 environment...")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # Wrap with RecordVideo
    env = RecordVideo(
        env, 
        video_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f'ppo_epsilon_{epsilon}'
    )
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent (with same config as training)
    print(f"Creating PPO agent (epsilon={epsilon})...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        clip_epsilon=epsilon
    )
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("\nIf you have trained agents in memory, you can save them using:")
        print("  from save_models import save_agent")
        print(f"  save_agent(your_agent, epsilon={epsilon})")
        print("\nOr check if models were saved in ./results/models/")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    agent.load(model_path)
    agent.policy.eval()
    
    print(f"\nRecording {num_episodes} episode(s)...")
    print("=" * 60)
    
    # Record episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        # Record full episodes - CartPole-v1 naturally terminates after 500 steps when solved
        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _, _ = agent.policy.get_action(state_tensor)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward
        
        episode_duration = steps * 0.02  # CartPole runs at ~50 FPS (0.02s per step)
        print(f"Episode {episode + 1}/{num_episodes}: {steps} steps, Return: {total_reward:.1f}, Duration: ~{episode_duration:.1f}s")
    
    env.close()
    print("=" * 60)
    print(f"Videos saved to {video_dir}")
    print(f"Look for files with prefix 'ppo_epsilon_{epsilon}'")


def main():
    parser = argparse.ArgumentParser(description='Record video of trained PPO agent')
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to saved model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--epsilon', 
        type=float, 
        default=0.2,
        help='Epsilon value used during training (default: 0.2)'
    )
    parser.add_argument(
        '--num_episodes', 
        type=int, 
        default=5,
        help='Number of episodes to record (default: 5 for longer videos)'
    )
    parser.add_argument(
        '--max_steps', 
        type=int, 
        default=1000,
        help='Maximum steps per episode (default: 1000)'
    )
    parser.add_argument(
        '--video_dir', 
        type=str, 
        default='./results/videos',
        help='Directory to save videos (default: ./results/videos)'
    )
    
    args = parser.parse_args()
    
    record_video_demo(
        model_path=args.model_path,
        epsilon=args.epsilon,
        num_episodes=args.num_episodes,
        video_dir=args.video_dir,
        max_steps_per_episode=args.max_steps
    )


if __name__ == '__main__':
    main()
