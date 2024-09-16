import torch
import torch.nn as nn
import torch.nn.functional as F



# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    discounted_rewards = torch.tensor(discounted_rewards)
    return discounted_rewards 


# Function to select action based on policy
def select_action(policy_net, state):
    state = torch.FloatTensor(state)
    action_probs = policy_net(state)
    action = torch.multinomial(action_probs, 1).item()  # Sample from the probability distribution
    return action, action_probs[action]




class Trainer():
    def __init__(self, model, optimizer, env, num_episodes=1000, gamma=0.99, batch_size=32):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model.to(self.device)



    def train(self):
        self.model.train()

        for episode in range(self.num_episodes):
            state = self.env.reset()[0]
            log_probs = []
            rewards = []

            done = False
            while not done:
                action, prob = select_action(self.model, state)
                next_state, reward, done, _, _ = self.env.step(action)

                log_probs.append(torch.log(prob))
                rewards.append(reward)

                state = next_state

                if done:
                    discounted_rewards = compute_discounted_rewards(rewards, self.gamma)
                    loss = 0
                    for log_prob, reward in zip(log_probs, discounted_rewards):
                        loss -= log_prob * reward  # Loss is negative log probability * reward (Policy Gradient)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if (episode+1)%20 == 0:

                        print(f"Episode {episode}, Total Reward: {sum(rewards)}")
                    break