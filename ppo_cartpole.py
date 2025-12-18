import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from collections import deque
import os


class ActorCritic(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor (2 hidden layers)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy network) - outputs logits for actions
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network) - outputs state value
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """Forward pass through the network."""
        features = self.feature_extractor(state)
        
        # Policy logits (will be converted to probabilities via softmax)
        logits = self.actor(features)
        
        # State value estimate
        value = self.critic(features)
        
        return logits, value
    
    def get_action(self, state):
        logits, value = self.forward(state)
        
        # Convert logits to probability distribution
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class PPOAgent:
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dim=64
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize policy network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Store old policy for importance sampling
        self.old_policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def compute_gae(self, rewards, values, next_value, dones):
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        # Start GAE computation from the end
        next_value_use = next_value
        gae = 0
        
        for t in reversed(range(T)):
            if dones[t]:
                # Terminal state: no bootstrap from next state
                delta = rewards[t] - values[t]
                gae = delta
                # For terminal states, next iteration should use 0 (no bootstrap)
                next_value_use = 0.0
            else:
                # Non-terminal: bootstrap from next state's value
                delta = rewards[t] + self.gamma * next_value_use - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                # Update next_value_use for next iteration (backwards in time)
                next_value_use = values[t]
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update_old_policy(self):
        """Update old policy to current policy."""
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def save(self, filepath):
        """Save the agent's policy network."""
        torch.save(self.policy.state_dict(), filepath)
        print(f"Saved model to {filepath}")
    
    def load(self, filepath):
        """Load the agent's policy network."""
        self.policy.load_state_dict(torch.load(filepath, map_location='cpu'))
        self.policy.eval()
        print(f"Loaded model from {filepath}")
    
    def train_step(self, states, actions, old_log_probs, advantages, returns, epochs=4, batch_size=64):
        self.policy.train()
        T = len(states)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(T)
            
            for start_idx in range(0, T, batch_size):
                end_idx = min(start_idx + batch_size, T)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions under current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Compute probability ratio: r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss (MSE)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Compute approximate KL divergence for monitoring
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl.item()
        
        num_updates = epochs * (T // batch_size + (1 if T % batch_size else 0))
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl': total_kl / num_updates
        }
    
    def collect_trajectory(self, env, max_steps=500):
        self.policy.eval()
        
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state, _ = env.reset()
        
        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob, value = self.policy.get_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            state = next_state
            
            if done:
                break
        
        # Compute next_value (0 if done, otherwise bootstrap)
        next_value = 0.0 if dones[-1] else self.policy(torch.FloatTensor(state).unsqueeze(0))[1].item()
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(
            np.array(rewards),
            np.array(values),
            next_value,
            np.array(dones)
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'returns': returns,
            'advantages': advantages,
            'log_probs': np.array(log_probs),
            'episode_return': sum(rewards),
            'values': np.array(values)  # Store value estimates for analysis
        }


def train_ppo(env, agent, total_steps=50000, update_frequency=2048, ppo_epochs=4, 
              batch_size=64, render=False, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    
    episode_returns = []
    episode_steps = []  # Track step indices for each episode
    kl_divergences = []
    policy_losses = []
    value_losses = []
    entropies = []
    avg_value_estimates = []
    steps = 0
    
    print(f"Training PPO with epsilon={agent.clip_epsilon}")
    print(f"Total steps: {total_steps}, Update frequency: {update_frequency}")
    
    while steps < total_steps:
        # Collect trajectories
        trajectories = []
        collected_steps = 0
        
        while collected_steps < update_frequency and steps < total_steps:
            traj = agent.collect_trajectory(env, max_steps=500)
            trajectories.append(traj)
            traj_length = len(traj['states'])
            collected_steps += traj_length
            steps += traj_length
            
            episode_returns.append(traj['episode_return'])
            episode_steps.append(steps)  # Record step at which episode completed
            
            if render:
                env.render()
        
        if len(trajectories) == 0:
            break
        
        # Concatenate trajectories
        all_states = np.concatenate([t['states'] for t in trajectories])
        all_actions = np.concatenate([t['actions'] for t in trajectories])
        all_old_log_probs = np.concatenate([t['log_probs'] for t in trajectories])
        all_advantages = np.concatenate([t['advantages'] for t in trajectories])
        all_returns = np.concatenate([t['returns'] for t in trajectories])
        
        # Update old policy
        agent.update_old_policy()
        
        # Perform PPO update
        stats = agent.train_step(
            all_states, all_actions, all_old_log_probs,
            all_advantages, all_returns,
            epochs=ppo_epochs, batch_size=batch_size
        )
        
        kl_divergences.append(stats['kl'])
        policy_losses.append(stats['policy_loss'])
        value_losses.append(stats['value_loss'])
        entropies.append(stats['entropy'])
        
        # Track average value estimates from trajectories
        avg_values = np.mean([np.mean(t['values']) if 'values' in t else 0 
                              for t in trajectories])
        avg_value_estimates.append(avg_values)
        
        # Print progress with more detailed metrics
        recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
        avg_return = np.mean(recent_returns)
        success_rate = np.mean([r >= 195.0 for r in recent_returns]) * 100  # CartPole solved if return >= 195
        print(f"Steps: {steps}/{total_steps}, "
              f"Recent avg return: {avg_return:.2f}, "
              f"Success rate: {success_rate:.1f}%, "
              f"KL: {stats['kl']:.4f}, "
              f"Entropy: {stats['entropy']:.4f}")
    
    return {
        'episode_returns': episode_returns,
        'episode_steps': episode_steps,
        'kl_divergences': kl_divergences,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies,
        'avg_value_estimates': avg_value_estimates
    }


def plot_training_curves(results_dict, save_path='./results/training_curves.png'):
    plt.figure(figsize=(12, 6))
    
    for epsilon, results in results_dict.items():
        returns = np.array(results['episode_returns'])
        steps = np.array(results['episode_steps'])
        
        # Smooth the curve using moving average
        window = 20
        if len(returns) > window:
            smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
            # Adjust step indices: moving average reduces length by (window-1)
            # Use steps corresponding to the end of each window
            smoothed_steps = steps[window-1:]
            plt.plot(smoothed_steps, smoothed, label=f'ε = {epsilon}', linewidth=2)
        else:
            plt.plot(steps, returns, label=f'ε = {epsilon}', linewidth=2)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episode Return', fontsize=12)
    plt.title('PPO Training: Episode Return vs Training Steps', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_kl_divergence(results_dict, save_path='./results/kl_divergence.png'):
    plt.figure(figsize=(12, 6))
    
    for epsilon, results in results_dict.items():
        kl_divs = results['kl_divergences']
        steps = np.arange(len(kl_divs))
        plt.plot(steps, kl_divs, label=f'ε = {epsilon}', linewidth=2, alpha=0.7)
    
    plt.xlabel('Update Step', fontsize=12)
    plt.ylabel('Approximate KL Divergence', fontsize=12)
    plt.title('PPO Training: KL Divergence vs Updates', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved KL divergence plot to {save_path}")
    plt.close()


def plot_additional_metrics(results_dict, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Value estimates over training
    plt.figure(figsize=(12, 6))
    for epsilon, results in results_dict.items():
        if 'avg_value_estimates' in results:
            values = results['avg_value_estimates']
            steps = np.arange(len(values))
            plt.plot(steps, values, label=f'ε = {epsilon}', linewidth=2, alpha=0.7)
    plt.xlabel('Update Step', fontsize=12)
    plt.ylabel('Average Value Estimate', fontsize=12)
    plt.title('PPO Training: Value Function Estimates vs Updates', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/value_estimates.png', dpi=300, bbox_inches='tight')
    print(f"Saved value estimates plot to {save_dir}/value_estimates.png")
    plt.close()
    
    # Plot 2: Policy entropy over training
    plt.figure(figsize=(12, 6))
    for epsilon, results in results_dict.items():
        if 'entropies' in results:
            entropies = results['entropies']
            steps = np.arange(len(entropies))
            plt.plot(steps, entropies, label=f'ε = {epsilon}', linewidth=2, alpha=0.7)
    plt.xlabel('Update Step', fontsize=12)
    plt.ylabel('Policy Entropy', fontsize=12)
    plt.title('PPO Training: Policy Entropy vs Updates', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/policy_entropy.png', dpi=300, bbox_inches='tight')
    print(f"Saved policy entropy plot to {save_dir}/policy_entropy.png")
    plt.close()
    
    # Plot 3: Loss curves (policy and value)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for epsilon, results in results_dict.items():
        if 'policy_losses' in results:
            policy_losses = results['policy_losses']
            steps = np.arange(len(policy_losses))
            ax1.plot(steps, policy_losses, label=f'ε = {epsilon}', linewidth=2, alpha=0.7)
        
        if 'value_losses' in results:
            value_losses = results['value_losses']
            steps = np.arange(len(value_losses))
            ax2.plot(steps, value_losses, label=f'ε = {epsilon}', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Update Step', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12)
    ax1.set_title('Policy Loss vs Updates', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Update Step', fontsize=12)
    ax2.set_ylabel('Value Loss', fontsize=12)
    ax2.set_title('Value Loss vs Updates', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved loss curves plot to {save_dir}/loss_curves.png")
    plt.close()
    
    # Plot 4: Success rate over training (CartPole solved if return >= 195)
    plt.figure(figsize=(12, 6))
    for epsilon, results in results_dict.items():
        returns = results['episode_returns']
        steps = results['episode_steps']
        
        # Compute success rate with moving window
        window = 20
        success_rates = []
        success_steps = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            success_rate = np.mean([r >= 195.0 for r in window_returns]) * 100
            success_rates.append(success_rate)
            success_steps.append(steps[i])
        
        if len(success_rates) > 0:
            plt.plot(success_steps, success_rates, label=f'ε = {epsilon}', linewidth=2, alpha=0.7)
    
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Solved (100%)')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('PPO Training: Success Rate (Return ≥ 195) vs Training Steps', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/success_rate.png', dpi=300, bbox_inches='tight')
    print(f"Saved success rate plot to {save_dir}/success_rate.png")
    plt.close()


def record_video(env, agent, video_dir='./results/videos', num_episodes=3):
    os.makedirs(video_dir, exist_ok=True)
    
    agent.policy.eval()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _, _ = agent.policy.get_action(state_tensor)
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        print(f"Recorded episode {episode + 1}/{num_episodes} ({steps} steps)")
    
    env.close()
    print(f"Videos saved to {video_dir}")


def main():
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Experiment configuration - increased steps for better convergence
    epsilon_values = [0.1, 0.2, 0.3, 0.5]
    total_steps = 100000  # Increased from 50k to 100k for better research results
    update_frequency = 2048
    ppo_epochs = 4
    batch_size = 64
    
    # Store results for each epsilon
    all_results = {}
    trained_agents = {}
    
    print("=" * 60)
    print("PPO CartPole Training Experiments")
    print("=" * 60)
    
    # Train with different epsilon values
    for epsilon in epsilon_values:
        print(f"\n{'=' * 60}")
        print(f"Training with epsilon = {epsilon}")
        print(f"{'=' * 60}")
        
        # Create environment
        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Create agent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            clip_epsilon=epsilon
        )
        
        # Train
        results = train_ppo(
            env=env,
            agent=agent,
            total_steps=total_steps,
            update_frequency=update_frequency,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            render=False
        )
        
        all_results[epsilon] = results
        trained_agents[epsilon] = agent
        
        # Save model
        os.makedirs('./results/models', exist_ok=True)
        model_path = f'./results/models/ppo_epsilon_{epsilon}.pth'
        agent.save(model_path)
        
        env.close()
   
    
    plot_training_curves(all_results, './results/training_curves.png')
    plot_kl_divergence(all_results, './results/kl_divergence.png')
    plot_additional_metrics(all_results, './results')
    
    print("Recording video demo (epsilon=0.2)...")
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RecordVideo(env, './results/videos', episode_trigger=lambda x: True)
    
    record_video(env, trained_agents[0.2], './results/videos', num_episodes=5)  # Record more episodes
    
    print("Experiments completed!")


if __name__ == '__main__':
    main()

