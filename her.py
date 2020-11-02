from utils import *


class HER:
    def __init__(self, env, off_policy_agent):
        self.env = env
        self.agent = off_policy_agent

    def run_strategy(self, k, strategy='future'):
        if strategy == 'future':
            self._future_strategy(k)

        elif strategy == 'episode':
            self._episode_strategy(k)

        elif strategy == 'random':
            self._random_strategy(k)

    def _future_strategy(self, k):
        state, action, _, next_state, done, info, achieved_goal = self.agent.short_memory.sample(
            batch_size=len(self.agent.short_memory),
            random_=False)
        # print('len of episode_memory: ', len(episode_memory))
        for step in range(len(self.agent.short_memory)):
            for i in range(k):
                # state, action, np.array([reward]), next_state, done, info, achieved_goal
                if step == len(self.agent.short_memory):
                    future = len(self.agent.short_memory) - 1
                else:
                    future = np.random.randint(step, len(self.agent.short_memory))
                # print('future =', future)
                recycle_state = state[step][0:-3]
                recycle_action = action[step]
                recycle_next_state = next_state[step][0:-3]
                recycle_done = done[step]
                recycle_info = info[step]
                recycle_achieved_goal = achieved_goal[step]
                recycle_goal = achieved_goal[future]
                recycle_reward = self.env.compute_reward(recycle_achieved_goal, recycle_goal, recycle_info)

                self.agent.long_memory.push(np.concatenate((recycle_state, recycle_goal)), recycle_action,
                                            recycle_reward,
                                            np.concatenate((recycle_next_state, recycle_goal)), recycle_done,
                                            recycle_info,
                                            recycle_achieved_goal)

    def _episode_strategy(self, k):
        pass

    def _random_strategy(self, k):
        pass
