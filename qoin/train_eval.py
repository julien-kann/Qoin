import numpy as np

from utils import get_state
from tqdm import tqdm


def train(agent, episode, data, ep_count=100, batch_size=32, obs_dim=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    obs = get_state(data=data,
                    t=0,
                    n_timesteps=obs_dim + 1)

    for t in tqdm(range(data_length),
                  total=data_length,
                  leave=True,
                  desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        next_obs = get_state(data=data,
                             t=t + 1,
                             n_timesteps=obs_dim + 1)

        action = agent.get_action(obs)

        if action == 1:
            print("We buying")
            agent.inventory.append(data[t])

        elif action == 2 and len(agent.inventory) > 0:
            print("We selling")
            price_vec = agent.inventory.pop(0)
            reward = data[t] - price_vec
            total_profit += reward

        else:
            print("Hold!!")
            pass

        done = (t == data_length - 1)
        agent.remember(obs, action=action, reward=reward, next_state=next_obs, done=done)

        if len(agent.memory) > batch_size:
            loss = agent.learn(batch_size)
            avg_loss.append(loss)

        obs = next_obs
    avg_loss = np.mean(np.array(avg_loss))

    return episode, ep_count, total_profit, avg_loss


def evaluate_model(agent, data, n_timesteps):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []

    obs = get_state(data, 0, n_timesteps + 1)

    for t in range(data_length):
        reward = 0
        next_obs = get_state(data, t + 1, n_timesteps=n_timesteps + 1)

        action = agent.act(obs, is_eval=True)
        # BUY
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))


        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta  # max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((obs, action, reward, next_obs, done))

        obs = next_obs
        if done:
            return total_profit, history


if __name__ == "__main__":
    import pandas as pd
    from agent import Agent
    agent = Agent(obs_dim=24,
                  action_dim=3)
    btc = pd.read_csv("../data/Bitstamp_BTCUSD_1h.csv", skiprows=1, index_col="date")
    train_data = btc["close"].to_list()
    ep_count = 2400
    for episode in range(1, ep_count+1):
        print(episode)
        trainer = train(agent,
                        episode=episode,
                        data=train_data,
                        ep_count=ep_count,
                        batch_size=64,
                        obs_dim=24)
