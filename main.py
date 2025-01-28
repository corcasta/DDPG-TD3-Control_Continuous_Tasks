import sys
import gym
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from utils import *

def main():
    env = gym.make('Pendulum-v0')
    
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
    agent = DDPGAgent(env)
    batch_size = 128
    rewards = []
    avg_rewards = []
    
    for episode in range(50):
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0
    
        for step in range(500):
            if episode >= 45:
                env.render()
            action = agent.get_action(state, ou_noise)
            new_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, new_state, done)
    
            if len(agent.memory) > batch_size:
                agent.train(batch_size)
    
            state = new_state
            episode_reward += reward
    
            if done:
                if episode == 0:
                    sys.stdout.write(
                        "episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                                 np.round(episode_reward, decimals=2),
                                                                                 "nan"))
                else:
                    sys.stdout.write(
                        "episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                                 np.round(episode_reward, decimals=2),
                                                                                 np.mean(rewards[-10:])))
                break
    
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == "__main__":
    main()
