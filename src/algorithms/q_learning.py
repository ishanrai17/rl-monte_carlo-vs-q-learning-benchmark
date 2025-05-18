from .abstract_rl_algorithm import AbstractRLAlgorithm

GAMMA = 1
ALPHA = 0.1
EPSILON = 0.99
DECAY_RATE = 0.999

class QLearning(AbstractRLAlgorithm):
    def __init__(self, env):
        super().__init__(env)
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.epsilon = EPSILON
        self.decay_rate = DECAY_RATE

    def _execute_episode(self, update=True):
        current_state, _ = self.env.reset()
        total_reward = 0
        complete = False

        while not complete:
            self.env.render()
            action = self.choose_action(current_state)
            next_state, reward, complete, step_count = self.env.perform(action)

            # Only used for tracking total reward
            total_reward += reward

            if update:
                self._update(current_state, next_state, action, complete, reward)

            current_state = next_state

        self.env.render()

        if self.episode_num % 1000 == 0:
            print(f"Episode {self.episode_num}: Total Reward: {total_reward}, Steps: {step_count} Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}")

        return total_reward

    def _update(self, current_state, next_state, action, complete, reward):
        current_q = self.Q[current_state][action]
        max_next_q = max(self.Q[next_state]) if not complete else 0
        target = reward + self.gamma * max_next_q

        # Using Temporal Difference (TD) update
        self.Q[current_state][action] += self.alpha * (target - current_q)
