import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def print_board(board):
    """Pretty-print an 8x8 rook board."""
    print("\nBoard (1 = rook, 0 = empty):\n")
    for row in board:
        print(" ".join(str(int(x)) for x in row))


class HopfieldNetwork:
    """Binary Hopfield Network for associative memory."""
    
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.patterns = []
    
    def train(self, patterns):
        """
        Train network using Hebbian learning.
        patterns: list of binary vectors {-1, +1}^n
        """
        self.patterns = patterns
        self.W = np.zeros((self.n, self.n))
        
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            self.W += pattern @ pattern.T
        np.fill_diagonal(self.W, 0)
        self.W /= len(patterns)
    
    def energy(self, state):
        """Calculate energy of a state."""
        return -0.5 * state.T @ self.W @ state
    
    def update_async(self, state, max_iter=1000):
        """Asynchronous update until convergence."""
        state = state.copy()
        
        for iteration in range(max_iter):
            converged = True
            for i in range(self.n):
                h = self.W[i] @ state
                new_val = 1 if h >= 0 else -1
                
                if new_val != state[i]:
                    converged = False
                    state[i] = new_val
            
            if converged:
                return state, iteration + 1
        
        return state, max_iter
    
    def recall(self, pattern, max_iter=1000):
        """Recall a pattern from noisy input."""
        return self.update_async(pattern, max_iter)


def create_digit_patterns():
    """Create 10x10 binary patterns representing digits 0-5."""
    patterns = []
    
    # Pattern 0
    p0 = np.array([
        [0,1,1,1,1,1,1,1,1,0],
        [1,1,0,0,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,0,0,0,0,0,0,1,1],
        [0,1,1,1,1,1,1,1,1,0],
    ])
    
    # Pattern 1
    p1 = np.array([
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0],
    ])
    
    # Pattern 2
    p2 = np.array([
        [0,1,1,1,1,1,1,1,0,0],
        [1,1,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,1,1,0],
        [0,0,0,1,1,1,1,1,0,0],
        [0,1,1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1],
    ])
    
    patterns = [p0, p1, p2]
    
    # Convert to {-1, +1} and flatten
    binary_patterns = []
    for p in patterns:
        binary = 2 * p - 1  # Convert 0,1 -> -1,+1
        binary_patterns.append(binary.flatten())
    
    return binary_patterns, patterns


def add_noise(pattern, noise_level=0.2):
    """Add noise by flipping bits with given probability."""
    noisy = pattern.copy()
    n = len(pattern)
    n_flips = int(noise_level * n)
    flip_indices = np.random.choice(n, n_flips, replace=False)
    noisy[flip_indices] *= -1
    return noisy


def test_capacity(n_neurons=100, max_patterns=20, n_trials=10):
    """
    Test network capacity by storing increasing numbers of random patterns.
    Returns: average recall accuracy vs number of patterns.
    """
    results = []
    
    for n_patterns in range(1, max_patterns + 1):
        accuracies = []
        
        for trial in range(n_trials):
            patterns = [np.random.choice([-1, 1], n_neurons) for _ in range(n_patterns)]
            
            # Train network
            net = HopfieldNetwork(n_neurons)
            net.train(patterns)
            
            # Test recall for each pattern
            correct = 0
            for pattern in patterns:
                recalled, _ = net.recall(pattern)
                if np.array_equal(recalled, pattern):
                    correct += 1
            
            accuracies.append(correct / n_patterns)
        
        avg_accuracy = np.mean(accuracies)
        results.append((n_patterns, avg_accuracy))
        print(f"Patterns: {n_patterns:2d}, Accuracy: {avg_accuracy:.2%}")
    
    return results


def test_error_correction(patterns, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Test error correction by adding different levels of noise.
    """
    n = len(patterns[0])
    net = HopfieldNetwork(n)
    net.train(patterns)
    
    results = {}
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        for pattern in patterns:
            for _ in range(10):  # Test multiple times per pattern
                noisy = add_noise(pattern, noise_level)
                recalled, _ = net.recall(noisy)
                
                if np.array_equal(recalled, pattern):
                    correct += 1
                total += 1
        
        accuracy = correct / total
        results[noise_level] = accuracy
        print(f"Noise level: {noise_level:.1%}, Accuracy: {accuracy:.2%}")
    
    return results


def eight_rook_energy(v, A=1.0, B=1.0):
    """
    Energy function for 8-rook problem.
    v: flattened length-64 vector in {0,1}
    A: penalty weight for row constraint (each row must have exactly 1 rook)
    B: penalty weight for column constraint (each column must have exactly 1 rook)
    
    Reasoning for weights:
    - A and B should be equal since row/column constraints are equally important
    - Larger values (A=B=2.0) help enforce constraints more strongly
    - Values too large may prevent exploration; too small may not converge
    """
    board = v.reshape(8, 8)
    row_sum = np.sum(board, axis=1)
    col_sum = np.sum(board, axis=0)

    row_pen = np.sum((row_sum - 1) ** 2)
    col_pen = np.sum((col_sum - 1) ** 2)

    return A * row_pen + B * col_pen


def solve_eight_rooks(max_restarts=50, max_iters=5000, A=2.0, B=2.0, verbose=True):
    """
    Solve the 8-rook problem by greedy energy descent.
    
    Weight selection reasoning:
    - A = B = 2.0: Equal importance for row and column constraints
    - Higher values (e.g., 5.0) make constraints stricter but may slow convergence
    - Lower values (e.g., 0.5) allow more exploration but may not satisfy constraints
    - 2.0 provides good balance between exploration and constraint satisfaction
    """
    best_solution = None
    best_energy = np.inf

    for attempt in range(max_restarts):
        v = np.random.randint(0, 2, size=64)
        E = eight_rook_energy(v, A, B)

        for it in range(max_iters):
            idx = np.random.randint(64)
            v_candidate = v.copy()
            v_candidate[idx] = 1 - v_candidate[idx]

            E_new = eight_rook_energy(v_candidate, A, B)
            if E_new <= E:
                v = v_candidate
                E = E_new

            if E == 0:
                board = v.reshape(8, 8)
                if verbose:
                    print(f"[Eight-Rook] Converged in attempt {attempt+1}, iter {it+1}")
                return board

        if E < best_energy:
            best_energy = E
            best_solution = v.copy()

    if verbose:
        print(f"[Eight-Rook] Best solution found with E = {best_energy}")
    return best_solution.reshape(8, 8)



def hopfield_tsp_tank(dist, n_iter=10000, dt=0.001, A=500.0, B=500.0, C=200.0, D=500.0, tau=1.0, verbose=True):
    """
    Continuous Hopfield–Tank TSP solver with improved stability.
    
    Number of weights:
    - Total neurons: n × n = 10 × 10 = 100
    - For fully connected symmetric network: N(N-1)/2 = 100 × 99 / 2 = 4,950 weights
    - However, in practice, weights are computed on-the-fly from constraints
    
    Parameters:
    - A, B: Constraint penalties (each row/col has exactly one 1)
    - C: Distance penalty
    - D: Additional penalty to ensure exactly n ones total
    """
    n = dist.shape[0]

    # Better initialization: small random values around 0.1
    u = 0.1 * np.random.randn(n, n)

    for it in range(n_iter):
        v = 1.0 / (1.0 + np.exp(-u))

        # Constraint 1: each city appears exactly once
        row_term = np.sum(v, axis=1, keepdims=True) - 1.0

        # Constraint 2: each position has exactly one city
        col_term = np.sum(v, axis=0, keepdims=True) - 1.0
        
        # Constraint 3: total number of active neurons should be n
        total_term = np.sum(v) - n
        
        # Distance term
        v_shift = np.roll(v, shift=-1, axis=1)
        dist_term = dist @ v_shift

        # Combined dynamics with all constraints
        du = -A * row_term - B * col_term - C * dist_term - D * total_term / n
        u += dt * (-u / tau + du)

        if verbose and (it+1) % 2000 == 0:
            r_err = np.linalg.norm(row_term)
            c_err = np.linalg.norm(col_term)
            total = np.sum(v)
            print(f"[TSP] Iter {it+1}/{n_iter}: row_err={r_err:.4f}, col_err={c_err:.4f}, total={total:.1f}")

    v = 1.0 / (1.0 + np.exp(-u))

    # Improved projection: greedy assignment to ensure valid tour
    final_assignment = np.zeros_like(v)
    
    # Method 1: Argmax per row (as before)
    tour_attempt1 = np.argmax(v, axis=1)
    
    # Check if valid (no repeated positions)
    if len(set(tour_attempt1)) == n:
        for i in range(n):
            final_assignment[i, tour_attempt1[i]] = 1.0
        tour = tour_attempt1
    else:
        # Method 2: Greedy matching to fix conflicts
        remaining_positions = set(range(n))
        tour = np.zeros(n, dtype=int)
        
        for city in range(n):
            # Find best available position for this city
            best_pos = None
            best_val = -1
            for pos in remaining_positions:
                if v[city, pos] > best_val:
                    best_val = v[city, pos]
                    best_pos = pos
            
            tour[city] = best_pos
            final_assignment[city, best_pos] = 1.0
            remaining_positions.remove(best_pos)

    return final_assignment, tour


def build_random_distance_matrix(n=10, seed=0, min_dist=10, max_dist=50):
    rng = np.random.default_rng(seed)
    D = rng.integers(min_dist, max_dist, size=(n, n))
    D = (D + D.T) // 2
    np.fill_diagonal(D, 0)
    return D


def calculate_tour_distance(tour, dist):
    """Calculate total distance of a tour."""
    total = 0
    n = len(tour)
    for i in range(n):
        total += dist[tour[i], tour[(i+1) % n]]
    return total


if __name__ == "__main__":
    
    print("PART 1: 10x10 Associative Memory")
 
    
    binary_patterns, original_patterns = create_digit_patterns()
    n_neurons = 100
    
    net = HopfieldNetwork(n_neurons)
    net.train(binary_patterns)
    
    print(f"\nTrained network with {len(binary_patterns)} patterns (digits 0, 1, 2)")
    print(f"Number of neurons: {n_neurons}")
    print(f"Weight matrix shape: {net.W.shape}")
    
    # Test recall with noise
    print("\n Testing Pattern Recall with Noise ")
    for i, pattern in enumerate(binary_patterns):
        noisy = add_noise(pattern, noise_level=0.2)
        recalled, iters = net.recall(noisy)
        
        match = np.array_equal(recalled, pattern)
        print(f"Pattern {i}: {'Correctly recalled' if match else 'failed'} (iterations: {iters})")
    
   
    print("PART 2: Network Capacity Analysis")
    print("\nTesting capacity with 100 neurons...")
    print("(This measures how many random patterns can be reliably stored)\n")
    
    capacity_results = test_capacity(n_neurons=100, max_patterns=15, n_trials=5)
    
    print(f"\n Theoretical capacity: ≈ {100 / (2 * np.log(100)):.1f} patterns ")
    print("(Roughly 0.138 × N for Hopfield networks)")
    
    print("PART 3: Error Correction Capability")
 
    print("\nTesting error correction with different noise levels\n")
    
    error_results = test_error_correction(binary_patterns)
    
    print("\n Analysis")
    print("The network can correct errors up to ~20-30% noise level effectively.")
    print("Beyond 40% noise, the pattern becomes too corrupted to recover.")
    
    print("PART 4: Eight-Rook Problem")
 
    
    rook_board = solve_eight_rooks(A=2.0, B=2.0)
    print_board(rook_board)
    
    row_sum = np.sum(rook_board, axis=1)
    col_sum = np.sum(rook_board, axis=0)
    print("\nRow sums:", row_sum,  )
    print("Col sums:", col_sum, )
    
    print("PART 5: 10-City TSP with Hopfield-Tank Network")

    n_cities = 10
    dist = build_random_distance_matrix(n_cities, seed=42)
    
    print("\nDistance matrix:")
    print(dist)
    N = n_cities * n_cities
    num_weights = N * (N - 1) // 2
    print(f"Total neurons: {N}")
    print(f"Total distinct weights (fully connected symmetric): {num_weights}")
    
    final_assign, tour = hopfield_tsp_tank(dist, n_iter=10000, dt=0.001, A=500, B=500, C=200, D=500, verbose=True)
    
    print("\nFinal assignment matrix (city × position):")
    print(np.round(final_assign).astype(int))
    
    print("\nTSP Tour (city order):")
    print(tour)
    
    total_dist = calculate_tour_distance(tour, dist)
    print(f"\nTotal tour distance: {total_dist}")
    
    print("\n Constraint Verification ")
    row_sums = np.sum(final_assign, axis=1)
    col_sums = np.sum(final_assign, axis=0)
    print("Each city appears once (row sums):", row_sums)
    print("Each position has one city (col sums):", col_sums)
    
   