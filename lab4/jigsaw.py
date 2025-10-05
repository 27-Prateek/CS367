import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

# configuration
MATFILE = "scrambled_lena.mat"
SOLVED_FILE = "jigsaw_after.png" # output filename for the final
TILES = 4            # number of blocks along one dimension (4x4 = 16 unique tiles)
ORIENTS = 8          # number of orientations (rotations + flips)
EDGE_WIDTH = 6       # number of pixels from the edge used for cost calculation
SEED = 457
T0 = 100000.0        # initial temperature
COOL = 0.999996      # slow geometric cooling rate
ITER = 3000000       # total iterations for deep search

# cost function weights 
GRADIENT_WEIGHT = 0.45 
CORNER_WEIGHT = 7.0    

random.seed(SEED)
np.random.seed(SEED)

# file loading and preprocessing
# loads image data from the MAT file.
def load_ascii_mat_file(filepath):
   
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Using placeholder.")
        return np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)

    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    dims = data_lines[0].split()
    rows, cols = int(dims[0]), int(dims[1])
    pixel_lines = data_lines[1:]
    
    all_pixels = []
    for line in pixel_lines:
        # Extend list with all integer values found on the line
        all_pixels.extend([int(v) for v in line.split() if v]) 

    pixels = np.array(all_pixels, dtype=np.uint8)
    
    # ensure correct size
    if pixels.size < rows * cols:
        temp = np.zeros(rows * cols, dtype=np.uint8)
        temp[:pixels.size] = pixels
        pixels = temp

    print(f"Successfully loaded image data: {rows}x{cols}. Total pieces: {TILES*TILES}.")
    return pixels.reshape(rows, cols)

# load image and determine tile size
img = load_ascii_mat_file(MATFILE)
n = img.shape[0]
assert n % TILES == 0, "Image not divisible by TILES"
tsz = n // TILES # tile size (e.g., 512 / 4 = 128)

# Define the 8 possible transformations (Rotations and Flips) (note: here took help from chat gpt)
flip0 = lambda x: x
flip1 = lambda x: np.rot90(x, 1)
flip2 = lambda x: np.rot90(x, 2)
flip3 = lambda x: np.rot90(x, 3)
flip4 = lambda x: np.fliplr(x)
flip5 = lambda x: np.flipud(x)
flip6 = lambda x: np.flipud(np.rot90(x, 1))
flip7 = lambda x: np.fliplr(np.rot90(x, 1))
xfm_choices = [flip0, flip1, flip2, flip3, flip4, flip5, flip6, flip7]

# Create the list of all unique piece-orientation combinations (16 tiles * 8 orientations = 128 total)
# This allows quick lookup via a single index (0-127).
pieces = []
for i in range(TILES):
    for j in range(TILES):
        tile = img[i*tsz:(i+1)*tsz, j*tsz:(j+1)*tsz] # Extract the original tile (0-15)
        for k in xfm_choices:
            pieces.append(k(tile).copy()) 

# below function Calculates the sum of absolute horizontal and vertical gradients
def edge_gradient(arr):
    
    g1 = np.abs(np.gradient(arr.astype(float), axis=1))
    g2 = np.abs(np.gradient(arr.astype(float), axis=0))
    return g1 + g2

# below line Pre-calculate gradients for all 128 possible tiles
grads = [edge_gradient(tile) for tile in pieces]

# core Logic for State Management

# Calculates the 0-127 index into the 'pieces' list for tile at position (i, j).
def get_piece_index(base_map, orientations, i, j):
    
    # Index = (Original_Tile_ID * ORIENTS) + Orientation_ID
    base_index = base_map[i, j]
    orientation = orientations[i, j]
    return base_index * ORIENTS + orientation

# Cost Function using separate base map and orientation map. This function calculates the mismatch cost across all internal boundaries.
def enhanced_mse_score(base_map, orientations, pieces, grads, num_blocks, edge_width):
    
    total_cost = 0
    
    # Check Horizontal Adjacency (Right edges)
    for i in range(num_blocks):
        for j in range(num_blocks - 1):
            idx1 = get_piece_index(base_map, orientations, i, j)
            idx2 = get_piece_index(base_map, orientations, i, j + 1)
            
            p1, p2 = pieces[idx1], pieces[idx2]
            g1, g2 = grads[idx1], grads[idx2]
            
            # Pixel MSE difference over EDGE_WIDTH
            total_cost += np.mean((p1[:, -edge_width:].astype(int) - p2[:, :edge_width].astype(int)) ** 2)
            
            # Gradient MSE difference over EDGE_WIDTH (Weighted)
            total_cost += GRADIENT_WEIGHT * np.mean((g1[:, -edge_width:] - g2[:, :edge_width]) ** 2)
            
            # Corner Difference Penalty (Weighted)
            corner_diff = (abs(int(p1[0, -1]) - int(p2[0, 0])) + abs(int(p1[-1, -1]) - int(p2[-1, 0])))
            total_cost += CORNER_WEIGHT * corner_diff

    # Check Vertical Adjacency (Bottom edges)
    for i in range(num_blocks - 1):
        for j in range(num_blocks):
            idx1 = get_piece_index(base_map, orientations, i, j)
            idx2 = get_piece_index(base_map, orientations, i + 1, j)
            
            p1, p2 = pieces[idx1], pieces[idx2]
            g1, g2 = grads[idx1], grads[idx2]
            
            # Pixel MSE difference over EDGE_WIDTH
            total_cost += np.mean((p1[-edge_width:, :].astype(int) - p2[:edge_width, :].astype(int)) ** 2)
            
            # Gradient MSE difference over EDGE_WIDTH (Weighted)
            total_cost += GRADIENT_WEIGHT * np.mean((g1[-edge_width:, :] - g2[:edge_width, :]) ** 2)
            
            # Corner Difference Penalty (Weighted)
            corner_diff = (abs(int(p1[-1, 0]) - int(p2[0, 0])) + abs(int(p1[-1, -1]) - int(p2[0, -1])))
            total_cost += CORNER_WEIGHT * corner_diff

    return total_cost

# Utility Functions

# Reconstructs and saves the image from the base map and orientation map
def show_grid(base_map, orientations, out, block_list, size):
    
    out_img = np.zeros((TILES*size, TILES*size), np.uint8)
    for i in range(TILES):
        for j in range(TILES):
            idx = get_piece_index(base_map, orientations, i, j)
            out_img[i*size:(i+1)*size, j*size:(j+1)*size] = block_list[idx]
            
    # Save the reconstructed image
    plt.imsave(out, out_img, cmap='gray')
    return out_img

def anneal(initial_base_map, initial_orientations):
    base_map = initial_base_map.copy()
    orientations = initial_orientations.copy()
    
    best_base_map = base_map.copy()
    best_orientations = orientations.copy()
    min_score = enhanced_mse_score(base_map, orientations, pieces, grads, TILES, EDGE_WIDTH)
    current_score = min_score
    T = T0

    best_score_history = []
    no_improve_steps = 0  # Steps since last improvement

    print(f"Initial Score (Randomized Start): {min_score:.1f}")

    intervals = ITER // 100 
    start_time = time.time()

    for t in range(1, ITER + 1):
        act = random.choices(['swap', 'rotate'], [0.70, 0.30])[0] 
        new_base_map = base_map.copy()
        new_orientations = orientations.copy()
        if act == 'swap':
            if random.random() < 0.7:
                a, b = np.random.randint(0, TILES, 2), np.random.randint(0, TILES, 2)
                new_base_map[a[0], a[1]], new_base_map[b[0], b[1]] = new_base_map[b[0], b[1]], new_base_map[a[0], a[1]]
                new_orientations[a[0], a[1]], new_orientations[b[0], b[1]] = new_orientations[b[0], b[1]], new_orientations[a[0], a[1]]
            else:
                c1, c2 = np.random.randint(0, TILES, 2)
                new_base_map[:, [c1, c2]] = new_base_map[:, [c2, c1]]
                new_orientations[:, [c1, c2]] = new_orientations[:, [c2, c1]]
        elif act == 'rotate':
            a = np.random.randint(0, TILES, 2)
            current_rot = new_orientations[a[0], a[1]]
            rot_change = np.random.choice([-3, -2, -1, 1, 2, 3])
            new_rot = (current_rot + rot_change) % ORIENTS
            new_orientations[a[0], a[1]] = new_rot

        new_score = enhanced_mse_score(new_base_map, new_orientations, pieces, grads, TILES, EDGE_WIDTH)
        
        if new_score < current_score or (T > 1e-16 and math.exp((current_score - new_score) / T) > random.random()):
            base_map = new_base_map
            orientations = new_orientations
            current_score = new_score

            # Check for improvement
            if current_score < min_score:
                best_base_map = base_map.copy()
                best_orientations = orientations.copy()
                min_score = current_score
                no_improve_steps = 0  # Reset since we improved
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        # Reset temperature if no improvement for too long
        if no_improve_steps > 700000:
            T = T0  # Reheat
            no_improve_steps = 0
            print(f"[RESET] Temperature reheated at iteration {t}, best score stagnated.")

        T *= COOL
        
        if t % intervals == 0:
            best_score_history.append(min_score)
            time_elapsed = time.time() - start_time
            print(f"Iter {t}/{ITER} | Temp: {T:.2e} | Best Score: {min_score:.1f} | Elapsed: {time_elapsed:.2f}s")

        if T < 1e-16:
            print(f"Temperature reached minimum, stopping early at {t}/{ITER}")
            break

    return best_base_map, best_orientations, min_score, best_score_history

    
    # Simulated Annealing loop using dual state arrays to enforce non-duplication.
    
    base_map = initial_base_map.copy()
    orientations = initial_orientations.copy()
    
    # Initialize best state and score
    best_base_map = base_map.copy()
    best_orientations = orientations.copy()
    min_score = enhanced_mse_score(base_map, orientations, pieces, grads, TILES, EDGE_WIDTH)
    current_score = min_score
    T = T0
    
    best_score_history = []
    
    print(f"Initial Score (Randomized Start): {min_score:.1f}")
    
    intervals = ITER // 100 
    start_time = time.time()
    
    for t in range(1, ITER + 1):
        
        # Action distribution: 70% Swap (Position), 30% Rotate (Orientation)
        act = random.choices(['swap', 'rotate'], [0.70, 0.30])[0] 

        new_base_map = base_map.copy()
        new_orientations = orientations.copy()
        
        if act == 'swap':
            # Swap: Changes the position of two unique base tiles.
            
            # 70% chance of swapping random tiles, 30% chance of swapping entire columns
            if random.random() < 0.7:
                # Swap two arbitrary tile positions
                a, b = np.random.randint(0, TILES, 2), np.random.randint(0, TILES, 2)
                
                # Swap the unique base tile indices AND their current orientations
                new_base_map[a[0], a[1]], new_base_map[b[0], b[1]] = new_base_map[b[0], b[1]], new_base_map[a[0], a[1]]
                new_orientations[a[0], a[1]], new_orientations[b[0], b[1]] = new_orientations[b[0], b[1]], new_orientations[a[0], a[1]]
            else:
                # Swap entire columns
                c1, c2 = np.random.randint(0, TILES, 2)
                new_base_map[:, [c1, c2]] = new_base_map[:, [c2, c1]]
                new_orientations[:, [c1, c2]] = new_orientations[:, [c2, c1]]

        elif act == 'rotate':
            # Rotate: Changes only the orientation of the tile at one position.
            a = np.random.randint(0, TILES, 2)
            current_rot = new_orientations[a[0], a[1]]
            
            # Apply a random rotation change 
            rot_change = np.random.choice([-3, -2, -1, 1, 2, 3])
            new_rot = (current_rot + rot_change) % ORIENTS
            
            # Update the piece orientation
            new_orientations[a[0], a[1]] = new_rot

        # Calculate score of the new state
        new_score = enhanced_mse_score(new_base_map, new_orientations, pieces, grads, TILES, EDGE_WIDTH)
        
        # Metropolis algorithm acceptance criterion
        if new_score < current_score or (T > 1e-16 and math.exp((current_score - new_score) / T) > random.random()):
            base_map = new_base_map
            orientations = new_orientations
            current_score = new_score
            
            # Track the absolute best score found
            if current_score < min_score:
                best_base_map = base_map.copy()
                best_orientations = orientations.copy()
                min_score = current_score
        
        T *= COOL # Geometric Cooling Schedule
        
        # Logging and history tracking
        if t % intervals == 0:
            best_score_history.append(min_score)
            time_elapsed = time.time() - start_time
            print(f"Iter {t}/{ITER} | Temp: {T:.2e} | Best Score: {min_score:.1f} | Elapsed: {time_elapsed:.2f}s")
            
        if T < 1e-16:
             print(f"Temperature reached minimum, stopping early at {t}/{ITER}")
             break

    return best_base_map, best_orientations, min_score, best_score_history

# main logiv Execution

if __name__ == "__main__":
    
    # Initialize the Base Map (unique tile indices 0 to 15, shuffled)
    initial_base_map = np.arange(TILES * TILES).reshape(TILES, TILES)
    np.random.shuffle(initial_base_map.flat)
    
    # Initialize the Orientations Map (random orientation 0 to 7 for each position)
    initial_orientations = np.random.randint(0, ORIENTS, size=(TILES, TILES))
    
    # Show the random start state (guaranteed to be non-duplicate)
    show_grid(initial_base_map, initial_orientations, 'jigsaw_random_start_fixed.png', pieces, tsz)

    print("\nSimulated Annealing in progress...")
    
    # Run Simulated Annealing
    soln_base, soln_orient, energy, score_history = anneal(initial_base_map, initial_orientations)
    
    # Reconstruct the best image found and save it
    show_grid(soln_base, soln_orient, SOLVED_FILE, pieces, tsz)
    print(f"\nAnnealing done. Final Best Score: {energy:.1f}")
    print(f"SUCCESS: Solved image saved as '{SOLVED_FILE}'.")
    