import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()

class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        num_action_candidates = self.num_simulated_paths*self.horizon
        self.action_candidates = np.array([self.env.action_space.sample() for i in range(num_action_candidates)]) # (num, action_dim)

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        obs = state*np.ones((self.num_simulated_paths, state.shape[0])) # (paths, obs_dim)
        observations = [] # [(paths, obs_dim), ...]
        actions = [] # [(paths, action_dim), ...]
        next_observations = [] # [(paths, obs_dim), ...]

        for i in range(self.horizon):
            # sample from action candidates (instead of calling env.action_space.sample() every iteration)
            random_idx = np.random.choice(self.action_candidates.shape[0], obs.shape[0], replace=False)
            action = self.action_candidates[random_idx]
            #action = np.array([self.env.action_space.sample() for i in range(self.num_simulated_paths)])
            observations += [obs]
            actions += [action]
            obs = self.dyn_model.predict(obs, action)
            next_observations += [obs]

        costs = trajectory_cost_fn(self.cost_fn, observations, actions, next_observations) # (paths, )

        return actions[0][np.argmin(costs)]
