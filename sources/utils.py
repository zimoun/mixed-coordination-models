import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def random_argmax(x):
    """Argmax operation, but if there are multiple maxima, return one randomly.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    b = np.flatnonzero(arg_maxes)
    choice = np.random.choice(b)
    return choice


def all_argmax(x):
    """Argmax operation, but if there are multiple maxima, return all.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    indices = np.flatnonzero(arg_maxes)
    return indices


def softmax(x, beta=2):
    """Compute the softmax function.
    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))


def rotation_matrix_2d(angle):
    """

    :param angle: In radians
    :return:
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

def to_agent_frame(object_location, agent_location, agent_direction):
    """Shift reference frame to agent's current location and direction.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    translate = np.array(object_location) - np.array(agent_location)
    rotation_mat = rotation_matrix_2d(agent_direction).T
    result = rotation_mat.dot(translate)
    return np.asarray(result).squeeze()


def get_relative_angle(angle_1, angle_2):
    """Return the smallest difference in angles between two angles (in degrees).
    """
    a = angle_1 - angle_2
    a = (a + 180) % 360 - 180
    return a


def angle_to_landmark(agent_location, landmark_centre, agent_orientation):
    """Get the relative direction to the landmark from the viewpoint of the

    :return:
    """
    relative_cue_pos = to_agent_frame(landmark_centre, agent_location, agent_orientation)
    angle = np.arctan2(relative_cue_pos[1], relative_cue_pos[0])
    return np.degrees(angle)

def generate_random_policy(env):
    """Generate a random policy assigning equal probability to all possible actions in each state.

    :param env: Environment object to be evaluated.
    :return: Nested list[state][action] giving the probabilities of choosing [action] when in [state].
    """
    random_policy = []
    for state in env.state_indices:
        possible_actions = env.get_possible_actions(state)
        if not possible_actions:
            random_policy.append([[]])
            continue
        rand_pol = [1 / len(possible_actions)] * len(possible_actions)
        random_policy.append(rand_pol)
    return random_policy

def value_iteration(env, theta=1e-4, gamma=.9):
    """Implement the  value iteration algorithm from dynamic programming to compute the optimal policy and corresponding
    state-value function for a given environment.

    :param env: Environment object to be evaluated.
    :param theta: Cutoff criterion for convergence of the optimal policy.
    :param gamma: Exponential discount factor for future rewards.
    :return: List containing the optimal policy, and array containing values for each state.
    """
    optimal_values = find_optimal_values(env, gamma=gamma, theta=theta)
    optimal_policy = optimal_policy_from_value(env, optimal_values, gamma)
    return optimal_policy, optimal_values
