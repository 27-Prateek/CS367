# We have taken reference from 3 matlab codes extracted from the gbike.zip available in the pratik sir's repo & also from here "https://github.com/habanoz/reinforcement-learning-an-introduction/blob/master/chapter04/jacks_car_rental.py" 
# for the following code.
import time
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Problem Parameters from the question
# Expectations for Poisson distribution
LOC2_EXPECTED_RETURN = 2
LOC2_EXPECTED_REQUEST = 4
LOC1_EXPECTED_RETURN = 3
LOC1_EXPECTED_REQUEST = 3

# Maximum number of bikes to consider in Poisson calculation
MAX_EXPECTED_UPDATE = 11

# One bike can be moved from location 1 to 2 for free, because "She is happy to shuttle one bike to the second location for free."
FREE_BIKES_MOVE_TO_LOC2 = 1
# Additional cost for parking more than 10 bikes
EXTRA_PARKING_COST = 4

# MDP Constants
BIKE_MOVE_COST = 2
RENT_PRICE = 10
MAX_BIKES = 20
MAX_BIKES_MOVED = 5
GAMMA = 0.9  # Discount factor given in the question

# Pre-calculating Poisson probabilities for efficiency
PMF_MATRIX_MAP = {
    expected: [poisson.pmf(occurred, expected) for occurred in range(MAX_BIKES + 1)]
    for expected in [LOC1_EXPECTED_REQUEST, LOC2_EXPECTED_REQUEST, LOC1_EXPECTED_RETURN, LOC2_EXPECTED_RETURN]
}
USE_EXPECTED_RETURNS = True

# Represents a state with the number of bikes at each location.
class State:
    
    def __init__(self, nLoc1, nLoc2):
        self.nLoc1 = nLoc1
        self.nLoc2 = nLoc2

# Represents a possible daily outcome for requests and returns.
class ProbableUpdate:
    def __init__(self, req1, preq1, req2, preq2, ret1, pret1, ret2, pret2):
        self.req1, self.preq1 = req1, preq1
        self.req2, self.preq2 = req2, preq2
        self.ret1, self.pret1 = ret1, pret1
        self.ret2, self.pret2 = ret2, pret2


# Calculates the expected return for a given state and action.
#     :param bikes_moved: Negative actions are towards location 1, positive towards location 2.
#     :param s: The current state (number of bikes at each location).
#     :param V: The current value function matrix.
#     :return: The expected return for this action.
def step(bikes_moved, s, V):
    
    
    # Start with the cost of moving bikes
    bikes_moved_to_charge = bikes_moved
    if bikes_moved > 0:  # Moving bikes to location 2
        # Account for the one free bike shuttle
        bikes_moved_to_charge = max(bikes_moved_to_charge - FREE_BIKES_MOVE_TO_LOC2, 0)
    
    action_return = -abs(bikes_moved_to_charge) * BIKE_MOVE_COST

    # Number of bikes at each location after moving
    nLoc1 = s.nLoc1 - bikes_moved
    nLoc2 = s.nLoc2 + bikes_moved

    # Incur extra parking cost if bikes > 10
    if nLoc1 > 10:
        action_return -= EXTRA_PARKING_COST
    if nLoc2 > 10:
        action_return -= EXTRA_PARKING_COST
    
    # Iterate over all possible combinations of requests and returns
    for u in states_updates():
        # Number of bikes rented is limited by availability
        req1 = min(u.req1, nLoc1)
        req2 = min(u.req2, nLoc2)
        
        # Probability of this specific request/return outcome
        prob = u.preq1 * u.preq2 * u.pret1 * u.pret2

        # New state after rentals and returns
        spNLoc1 = min(nLoc1 - req1 + u.ret1, MAX_BIKES)
        spNLoc2 = min(nLoc2 - req2 + u.ret2, MAX_BIKES)

        # The reward includes rental income and the discounted value of the next state
        reward = (req1 + req2) * RENT_PRICE
        action_return += prob * (reward + GAMMA * V[spNLoc1, spNLoc2])

    return action_return

# Generator to iterate through all possible states.
def states():
    
    for nLoc1 in range(MAX_BIKES + 1):
        for nLoc2 in range(MAX_BIKES + 1):
            yield State(nLoc1, nLoc2)

# Generator for possible request/return updates.
def states_updates():
    
    for req1 in range(MAX_EXPECTED_UPDATE):
        preq1 = PMF_MATRIX_MAP.get(LOC1_EXPECTED_REQUEST)[req1]
        for req2 in range(MAX_EXPECTED_UPDATE):
            preq2 = PMF_MATRIX_MAP.get(LOC2_EXPECTED_REQUEST)[req2]
            # Using expected returns for simplicity as per the original problem
            if USE_EXPECTED_RETURNS:
                yield ProbableUpdate(req1, preq1, req2, preq2, LOC1_EXPECTED_RETURN, 1.0, LOC2_EXPECTED_RETURN, 1.0)
            else:
                # This part would iterate over return probabilities too if needed
                for ret1 in range(MAX_EXPECTED_UPDATE):
                    pret1 = PMF_MATRIX_MAP.get(LOC1_EXPECTED_RETURN)[ret1]
                    for ret2 in range(MAX_EXPECTED_UPDATE):
                        pret2 = PMF_MATRIX_MAP.get(LOC2_EXPECTED_RETURN)[ret2]
                        yield ProbableUpdate(req1, preq1, req2, preq2, ret1, pret1, ret2, pret2)

# Policy evaluation function.
def policy_evaluation(V, pi):
    
    print("Policy evaluation started...")
    theta = 1e-4  # Convergence threshold
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        istart = time.time()
        for s in states():
            v = V[s.nLoc1, s.nLoc2]
            action = pi[s.nLoc1, s.nLoc2]
            action_return = step(action, s, V)
            V[s.nLoc1, s.nLoc2] = action_return
            delta = max(delta, abs(v - action_return))
        
        print(f"Policy evaluation iteration {iteration}, max delta='{delta:.4f}' in {time.time() - istart:.2f} seconds")
        if delta < theta:
            break
    print("Policy evaluation completed.")

# Policy improvement function.
def policy_improvement(V, pi):
    
    print("Policy improvement started...")
    start = time.time()
    policy_stable = True
    for s in states():
        old_action = pi[s.nLoc1, s.nLoc2]
        actions = np.arange(-MAX_BIKES_MOVED, MAX_BIKES_MOVED + 1)
        action_returns = []
        for a in actions:
            # Action is valid only if enough bikes are available to move
            if (a >= 0 and s.nLoc1 >= a) or (a < 0 and s.nLoc2 >= -a):
                action_return = step(a, s, V)
                action_returns.append(action_return)
            else:
                action_returns.append(-1e9)  # Invalid actions are heavily penalized
        
        new_action = actions[np.argmax(action_returns)]
        pi[s.nLoc1, s.nLoc2] = new_action
        if new_action != old_action:
            policy_stable = False
            
    print(f"Policy improvement completed in {time.time() - start:.2f} seconds.")
    return policy_stable

# Plotting functions.
def plot_policy(ax, iteration, pi):

    ax.set_title(f'π$_{iteration}$', fontsize=20)
    
    x_vals = np.arange(MAX_BIKES + 1)
    y_vals = np.arange(MAX_BIKES + 1)
    X, Y = np.meshgrid(x_vals, y_vals)

    # The policy matrix pi is indexed [loc1, loc2], which corresponds to (y, x)
    # The contour plot will correctly map this.
    contours = ax.contour(X, Y, pi.T, levels=range(-MAX_BIKES_MOVED, MAX_BIKES_MOVED + 1), colors='k')
    ax.clabel(contours, inline=True, fontsize=12, fmt='%d')
    
    ax.set_xlabel('# Bikes at second location', fontsize=14)
    ax.set_ylabel('# Bikes at first location', fontsize=14)
    ax.set_ylim(bottom=-0.5, top=MAX_BIKES + 0.5)
    ax.set_xlim(left=-0.5, right=MAX_BIKES + 0.5)

# Plots the final value function as a heatmap.
def plot_value_function(ax, V, iteration):
    ax.set_title(f'v$_{iteration}$', fontsize=20)
    
    x_vals = np.arange(MAX_BIKES + 1)
    y_vals = np.arange(MAX_BIKES + 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # The value matrix V is indexed [loc1, loc2], which corresponds to (y, x)
    ax.plot_surface(X, Y, V.T, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('# Bikes at second location', fontsize=12, labelpad=10)
    ax.set_ylabel('# Bikes at first location', fontsize=12, labelpad=10)
    ax.set_zlabel('Value', fontsize=12, labelpad=10)

# Main function to run the policy iteration algorithm.
def policy_iteration(file_name='gbike_solution.png'):
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    pi = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

    fig = plt.figure(figsize=(24, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(5)]
    ax_3d = fig.add_subplot(2, 3, 6, projection='3d')

    plot_policy(axes[0], 0, pi)

    iteration = 0
    while iteration < 4: # Limit to 4 iterations to match the book's figure
        iteration += 1
        policy_evaluation(V, pi)
        policy_stable = policy_improvement(V, pi)
        plot_policy(axes[iteration], iteration, pi)
        print(f"Policy iteration {iteration} completed.")
        if policy_stable: 
            # If it converges early, hide unused plot axes
            for i in range(iteration + 1, 5):
                axes[i].axis('off')
            break
            
    plot_value_function(ax_3d, V, iteration)
    
    plt.savefig(file_name)
    plt.close()
    print(f"Solution saved to {file_name}")

if __name__ == '__main__':

    policy_iteration()
