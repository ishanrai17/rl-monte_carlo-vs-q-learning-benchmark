from .abstract_rl_algorithm import AbstractRLAlgorithm

class QLearning(AbstractRLAlgorithm):
    def _execute_episode(self, update=True):
        current_state, _ = self.env.reset()
        total_reward = 0
        complete = False
        step_count = 0

        while not complete and step_count < self.env.maximum_steps:
            self.env.render()
            action = self.choose_action(current_state)
            next_state, reward, complete = self.env.perform(action)

            # Only used for tracking total reward
            total_reward += reward

            if update:
                self._update(current_state, next_state, action, complete, reward)

            current_state = next_state
            step_count += 1

        self.env.render()

        if self.episode_num % 100 == 0:
            print(f"Episode {self.episode_num}: Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}")

    def _update(self, current_state, next_state, action, complete, reward):
        current_q = self.Q[current_state][action]
        max_next_q = max(self.Q[next_state]) if not complete else 0
        target = reward + self.gamma * max_next_q

        # Using Temporal Difference (TD) update
        self.Q[current_state][action] += self.alpha * (target - current_q)
