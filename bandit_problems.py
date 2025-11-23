# Lab 7: Reinforcement Learning - MENACE and Bandits
# Student implementation

import numpy as np
import matplotlib.pyplot as plt
import random


# Part 1: MENACE for Tic-Tac-Toe


class MENACE:
    # Machine learning for noughts and crosses using matchbox analogy
    
    def __init__(self, start_beads=10):
        self.boxes = {}  # stores game states
        self.start_beads = start_beads
        self.history = []
        
    def state_to_str(self, b):
        return ''.join(str(x) for x in b)
    
    def valid_positions(self, b):
        return [i for i, val in enumerate(b) if val == 0]
    
    def init_state(self, b):
        s = self.state_to_str(b)
        if s not in self.boxes:
            counts = [self.start_beads if b[i] == 0 else 0 for i in range(9)]
            self.boxes[s] = counts
    
    def select_move(self, b):
        # Choose position weighted by bead count
        s = self.state_to_str(b)
        self.init_state(b)
        
        counts = self.boxes[s]
        valid = self.valid_positions(b)
        
        total = sum([counts[i] for i in valid])
        if total == 0:
            return random.choice(valid)
        
        wts = [counts[i] for i in valid]
        pos = random.choices(valid, weights=wts)[0]
        self.history.append((s, pos))
        return pos
    
    def learn(self, outcome):
        # Update beads based on game result
        for s, pos in self.history:
            if outcome > 0:
                self.boxes[s][pos] += 3
            elif outcome == 0:
                self.boxes[s][pos] += 1
            else:
                self.boxes[s][pos] = max(0, self.boxes[s][pos] - 1)
        self.history = []
    
    def clear_history(self):
        self.history = []
    
    def stats(self):
        return {
            'states': len(self.boxes),
            'beads': sum(sum(c) for c in self.boxes.values())
        }

def random_player(b):
    valid = [i for i, val in enumerate(b) if val == 0]
    return random.choice(valid) if valid else None

def check_win(b):
    # Check all winning patterns
    patterns = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
    for p in patterns:
        if b[p[0]] == b[p[1]] == b[p[2]] != 0:
            return b[p[0]]
    return 0

def play_game(agent, opponent):
    b = [0] * 9
    agent.clear_history()
    
    for turn in range(9):
        if turn % 2 == 0:  # agent turn
            pos = agent.select_move(b)
            b[pos] = 1
            if check_win(b) == 1:
                agent.learn(1)
                return 1
        else:  # opponent turn
            pos = opponent(b)
            if pos is None:
                break
            b[pos] = 2
            if check_win(b) == 2:
                agent.learn(-1)
                return -1
    
    agent.learn(0)
    return 0

def run_menace(games=1000):
    agent = MENACE(start_beads=10)
    res = {'w': 0, 'l': 0, 'd': 0}
    
    for i in range(games):
        outcome = play_game(agent, random_player)
        if outcome == 1:
            res['w'] += 1
        elif outcome == -1:
            res['l'] += 1
        else:
            res['d'] += 1
    
    return res, agent.stats()


# Part 2: Binary Bandits


class BinaryBandit:
    def __init__(self, p):
        self.p = p
    
    def pull(self):
        return 1 if np.random.random() < self.p else 0

def eps_greedy_binary(b1, b2, eps=0.1, steps=10000):
    # Epsilon-greedy strategy for two bandits
    q = np.zeros(2)
    n = np.zeros(2)
    
    r_list = []
    a_list = []
    q_hist = [q.copy()]
    
    for t in range(steps):
        # Select action
        if np.random.random() < eps:
            a = np.random.randint(2)
        else:
            a = np.argmax(q)
        
        # Get reward
        r = b1.pull() if a == 0 else b2.pull()
        
        # Update Q
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        
        r_list.append(r)
        a_list.append(a)
        q_hist.append(q.copy())
    
    return np.array(r_list), np.array(a_list), np.array(q_hist)


# Part 3: Non-stationary Bandit


class NonStatBandit:
    # 10-arm bandit with drifting rewards
    
    def __init__(self, arms=10, init_mu=0.0):
        self.arms = arms
        self.mu = np.ones(arms) * init_mu
        self.sigma = 0.01
        self.count = 0
    
    def pull(self, a):
        # Get reward and update means
        r = np.random.normal(self.mu[a], 1.0)
        self.mu += np.random.normal(0, self.sigma, self.arms)
        self.count += 1
        return r
    
    def best_arm(self):
        return np.argmax(self.mu)
    
    def get_means(self):
        return self.mu.copy()


# Part 4: Modified Epsilon-Greedy


def eps_greedy_nonstat(bandit, eps=0.1, alpha=0.1, steps=10000):
    # Uses constant step-size for non-stationary case
    
    q = np.zeros(bandit.arms)
    n = np.zeros(bandit.arms)
    
    r_list = []
    opt_list = []
    act_list = []
    q_hist = [q.copy()]
    
    for t in range(steps):
        # Select
        if np.random.random() < eps:
            a = np.random.randint(bandit.arms)
        else:
            a = np.argmax(q)
        
        # Execute
        r = bandit.pull(a)
        n[a] += 1
        
        # Update with constant alpha
        q[a] += alpha * (r - q[a])
        
        # Track
        r_list.append(r)
        opt = bandit.best_arm()
        opt_list.append(1 if a == opt else 0)
        act_list.append(a)
        q_hist.append(q.copy())
    
    return {
        'r': np.array(r_list),
        'opt': np.array(opt_list),
        'act': np.array(act_list),
        'qh': np.array(q_hist),
        'n': n
    }

def compare_methods(runs=100, steps=10000):
    # Compare sample average vs constant step-size
    
    alphas = [0.1, 0.2, 0.5]
    data = {
        'avg': np.zeros((runs, steps)),
        'a01': np.zeros((runs, steps)),
        'a02': np.zeros((runs, steps)),
        'a05': np.zeros((runs, steps))
    }
    
    for run in range(runs):
        # Create separate bandits
        b_avg = NonStatBandit(arms=10)
        b_01 = NonStatBandit(arms=10)
        b_02 = NonStatBandit(arms=10)
        b_05 = NonStatBandit(arms=10)
        
        # Initialize
        q_avg = np.zeros(10)
        n_avg = np.zeros(10)
        q_01 = np.zeros(10)
        q_02 = np.zeros(10)
        q_05 = np.zeros(10)
        
        for t in range(steps):
            # Sample average
            a = np.argmax(q_avg) if np.random.random() > 0.1 else np.random.randint(10)
            r = b_avg.pull(a)
            n_avg[a] += 1
            q_avg[a] += (r - q_avg[a]) / n_avg[a]
            data['avg'][run, t] = (1 if a == b_avg.best_arm() else 0)
            
            # Alpha 0.1
            a = np.argmax(q_01) if np.random.random() > 0.1 else np.random.randint(10)
            r = b_01.pull(a)
            q_01[a] += 0.1 * (r - q_01[a])
            data['a01'][run, t] = (1 if a == b_01.best_arm() else 0)
            
            # Alpha 0.2
            a = np.argmax(q_02) if np.random.random() > 0.1 else np.random.randint(10)
            r = b_02.pull(a)
            q_02[a] += 0.2 * (r - q_02[a])
            data['a02'][run, t] = (1 if a == b_02.best_arm() else 0)
            
            # Alpha 0.5
            a = np.argmax(q_05) if np.random.random() > 0.1 else np.random.randint(10)
            r = b_05.pull(a)
            q_05[a] += 0.5 * (r - q_05[a])
            data['a05'][run, t] = (1 if a == b_05.best_arm() else 0)
    
    return data


# Plotting functions


def plot_binary(r, a, qh, probs):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    
    ax[0].plot(np.cumsum(r))
    ax[0].set_title('Cumulative Reward')
    ax[0].set_xlabel('Steps')
    ax[0].grid(True)
    
    ax[1].plot(qh[:, 0], label=f'Q1 -> {probs[0]:.2f}')
    ax[1].plot(qh[:, 1], label=f'Q2 -> {probs[1]:.2f}')
    ax[1].axhline(probs[0], color='blue', linestyle='--', alpha=0.3)
    ax[1].axhline(probs[1], color='orange', linestyle='--', alpha=0.3)
    ax[1].set_title('Q-value Evolution')
    ax[1].legend()
    ax[1].grid(True)
    
    freq = [np.sum(a == 0), np.sum(a == 1)]
    ax[2].bar(['Bandit 1', 'Bandit 2'], freq)
    ax[2].set_title('Selection Frequency')
    ax[2].grid(True, axis='y')
    
    plt.tight_layout()
    return fig

def plot_nonstat(res, win=100):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    
    avg_r = np.convolve(res['r'], np.ones(win)/win, mode='valid')
    ax[0].plot(avg_r)
    ax[0].set_title(f'Reward (window={win})')
    ax[0].grid(True)
    
    opt_pct = np.convolve(res['opt'].astype(float), np.ones(win)/win, mode='valid') * 100
    ax[1].plot(opt_pct)
    ax[1].set_title('Optimal Selection %')
    ax[1].axhline(50, color='r', linestyle='--', alpha=0.3)
    ax[1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_comparison(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for method, vals in data.items():
        avg = np.mean(vals, axis=0) * 100
        smooth = np.convolve(avg, np.ones(100)/100, mode='valid')
        ax.plot(smooth, label=method, linewidth=2)
    
    ax.set_title('Method Comparison')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Optimal %')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


# Main


if __name__ == "__main__":
    print("="*60)
    print("Lab 7: RL Experiments")
    print("="*60)
    
    # Part 1
    print("\nPart 1: MENACE")
    res, st = run_menace(games=1000)
    print(f"W:{res['w']} D:{res['d']} L:{res['l']}")
    print(f"States: {st['states']}, Beads: {st['beads']}")
    
    # Part 2
    print("\nPart 2: Binary Bandits")
    b1 = BinaryBandit(0.3)
    b2 = BinaryBandit(0.7)
    r, a, qh = eps_greedy_binary(b1, b2, eps=0.1, steps=10000)
    print(f"Final Q: [{qh[-1,0]:.3f}, {qh[-1,1]:.3f}]")
    print(f"Total reward: {np.sum(r):.0f}")
    print(f"Avg reward: {np.mean(r):.3f}")
    
    fig1 = plot_binary(r, a, qh, [0.3, 0.7])
    plt.savefig('binary_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Part 3 & 4
    print("\nPart 3-4: Non-stationary Bandit")
    bandit = NonStatBandit(arms=10)
    res = eps_greedy_nonstat(bandit, eps=0.1, alpha=0.1, steps=10000)
    print(f"Avg reward: {np.mean(res['r']):.3f}")
    print(f"Optimal: {np.mean(res['opt'])*100:.1f}%")
    
    print("\nLast 1000 steps:")
    print(f"  Reward: {np.mean(res['r'][-1000:]):.3f}")
    print(f"  Optimal: {np.mean(res['opt'][-1000:])*100:.1f}%")
    
    fig2 = plot_nonstat(res, win=100)
    plt.savefig('nonstat_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Comparison
    print("\nComparing methods...")
    comp = compare_methods(runs=50, steps=10000)
    
    print("Final performance:")
    for m, d in comp.items():
        perf = np.mean(d[:, -1000:]) * 100
        print(f"  {m}: {perf:.1f}%")
    
    fig3 = plot_comparison(comp)
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Completed")
    print("="*60)
