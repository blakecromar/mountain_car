import sys
import gym
import numpy as np

import matplotlib.pyplot as plt

import warnings

# Ignore warnings
warnings.filterwarnings("ignore") 

# Set plotting options
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.
    The bins are spaces between the low and high values provided. The grid
    doesn't include the low and high values themselves.
    
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """  
    
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] 
            for dim in range(len(bins))]

    return grid 

def discretize(sample, grid):
    """Discretize a sample as per given grid.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by 
    discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, 
                 seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        # state_size = n-dimensional state space
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
         # 1-dimensional discrete action space
        self.action_size = self.env.action_space.n 
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        # initial exploration rate
        self.epsilon = self.initial_epsilon = epsilon  
        # how quickly should we decrease epsilon
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test').
        """
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: The Q table entry is updated for the *last* (state, action) 
            # pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += (self.alpha 
                 * (reward + self.gamma * max(self.q_table[state]) - 
                 self.q_table[self.last_state + (self.last_action,)]))

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

def run(agent, env, num_episodes=50000, mode='train'):
    """Run agent in given reinforcement learning environment and return 
    scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(
                    i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores

def plot_q_table(q_table, bins):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=bins)
    cax = ax.imshow(q_image, cmap='jet');
    _ = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')

def render_agent(q_agent, env): 
    """ A visual representation of the AI agent playing the Mountain Car game
    Parameters
    ----------
    q_agent : Object
        An AI agent that knows how to play the Mountain Car game
    env : Object
        The Mountain Car game created by OpenAI

    Returns
    -------
    None.

    """
    
    state = env.reset()
    score = 0
    for t in range(200):
        action = q_agent.act(state, mode='test')
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    print('Final score:', score)

def main():
    """Run the functions required to train the AI agent to play the mountain
    car game. It also runs functions to plot the Q-table along with playing
    the game live.

    Returns
    -------
    None.

    """
    # Create an environment and set random seed
    env = gym.make('MountainCar-v0')
    env.seed(505)   
    
    # Lower and higher bounds of dimensions
    low = env.observation_space.low
    high = env.observation_space.high
    
    # Creating grid
    state_grid = create_uniform_grid(low, high, bins=(20, 20))
    
    # Create agent
    q_agent = QLearningAgent(env, state_grid)
    
    # Train agent
    run(q_agent, env)
    
    # Plot Q Table
    plot_q_table(q_agent.q_table, bins=(20, 20))
    
    # Visualize the agent run
    for _ in range(1000):
        render_agent(q_agent, env)
    
if __name__ == "__main__":
    main()
    


    
    