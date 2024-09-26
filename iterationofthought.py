import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Iteration of Thought (IoT) with Planning and DPO
class IterationOfThought:
    def __init__(self, policy_network, reward_model, max_iterations=5, gamma=0.99):
        self.policy_network = policy_network
        self.reward_model = reward_model
        self.max_iterations = max_iterations
        self.gamma = gamma  # Discount factor

    def run(self, input_prompt):
        state_sequence = []
        trajectory = []
        total_reward = 0.0

        for t in range(self.max_iterations):
            # Generate action
            action_text, input_ids, attention_mask = self.policy_network.generate_action(input_prompt)
            logger.info(f"Iteration {t}: Action - {action_text}")

            # Update state
            state = input_prompt + action_text
            state_sequence.append(state)

            # Get reward
            reward = self.reward_model.get_reward(state)
            logger.info(f"Iteration {t}: Reward - {reward}")

            # Accumulate discounted reward
            total_reward += (self.gamma ** t) * reward

            # Store in trajectory
            trajectory.append((state, action_text, reward))

            # Check for termination condition
            if "Final Answer:" in action_text:
                break
            else:
                # Update input prompt for next iteration
                input_prompt = state

        return state_sequence, trajectory, total_reward
