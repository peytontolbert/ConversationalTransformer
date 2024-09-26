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
            # Edit 1: Generate thought and obtain student_logits
            thought_text, student_logits, input_ids, attention_mask = self.policy_network.generate_thought(input_prompt)  # {{ edit }}
            thought_text = thought_text.strip()
            if not thought_text:
                thought_text = "<No Thought Generated>"  # {{ edit_1 }}
            logger.info(f"Iteration {t}: Thought - '{thought_text}'")  # {{ edit_2 }}

            # Update state
            state = input_prompt + " " + thought_text
            state_sequence.append(state)

            # Edit 3: Pass student_logits to get_reward
            reward = self.reward_model.get_reward(state, student_logits)  # {{ edit_3 }}
            logger.info(f"Iteration {t}: Reward - {reward}")  # {{ edit_4 }}

            # Accumulate discounted reward
            total_reward += (self.gamma ** t) * reward

            # Store in trajectory
            trajectory.append((state, thought_text, reward))

            # Check for termination condition
            if "Final Answer:" in thought_text or "Final Answer" in thought_text:
                logger.info(f"Termination condition met at iteration {t}")  # {{ edit_5 }}
                break
            else:
                # Update input prompt for next iteration
                input_prompt = state
                logger.debug(f"Updated input_prompt for next iteration: '{input_prompt}'")  # {{ edit_6 }}

        return state_sequence, trajectory, total_reward
