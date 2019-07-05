from DDPG_Agent import DDPGAgent
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Initialize the agent
agents = DDPGAgent(state_size=24, action_size=2, seed=1, num_agents=num_agents)

n_episodes = 5000
scores = []
scores_window = deque(maxlen=100)
max_steps = 1000

for episode in range(n_episodes):
    env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
    states = env_info.vector_observations
    agents.reset()

    score = np.zeros(num_agents)

    for i in range(max_steps):
        actions = agents.act(states)
        env_info = env.step(actions)[brain_name]               # send the action to the environment
        next_states = env_info.vector_observations               # get the next state
        rewards = env_info.rewards                               # get the reward
        dones = env_info.local_done                              # see if episode has finished

        agents.step(states, actions, rewards, next_states, dones)

        score += rewards                                         # update the score

        states = next_states                                     # roll over the state to next time step

        if np.any(dones):                                          # exit loop if episode finished
            break
    agents.checkpoint()                                              # Save network weights
    # print("Episode Done")

    scores.append(np.mean(score))
    scores_window.append(np.mean(score))

    print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score),
                                                                               np.mean(scores_window)), end="")

    # Stop training if average score for last 100 episodes is more than or equal to 30
    if np.mean(scores_window) >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode+1, np.mean(scores_window)))
        break

# Close the environment
env.close()

# Plot scores against number of episodes
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

