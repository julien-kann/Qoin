import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import FeedForwardNN
import numpy as np
from torch.optim import Adam


class PPO:
    def __init__(self, env):
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters
        self._init_hyperparameters()

        # extract environmental information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG Step 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        self._init_hyperparameters()

        # Intialize actor optimizer
        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        # create covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=.5)
        self.cov_mat = torch.diag(self.cov_var
                                  )

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calc timesteps in batch
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calc ratio
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calc surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, + self.clip) * A_k

                # Calc actor and critic loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # calc gradients and perform backward propagation for actor netowrk
                self.actor_opt.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_opt.step()

                # Calculate gradients and perform back prop on critic netowrk
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate log probabilities of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # return predicted values V and log_probs
        return V, log_probs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()[0]
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Timesteps in batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                ok = self.env.step(action)
                obs, rew, done, _ = self.env.step(action)[:4]

                # Collect reward, action and log_probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # Compute rewards
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # Rewards to go per episode per patch to return
        batch_rtgs = []
        # Iterate backwards to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs):
        # Query actor network for a mean action
        mean = self.actor(obs)

        # Get multivariate normal dist with cov mat
        dist = MultivariateNormal(mean, self.cov_mat)

        # SAmple an action form distribution and get its log probability
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return sampled action and log prob of action
        return action.detach().numpy(), log_prob.detach()


if __name__ == "__main__":
    import gym

    env = gym.make('Pendulum-v1')
    model = PPO(env)
    model.learn(10000)
