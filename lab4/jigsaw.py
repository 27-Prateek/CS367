import numpy as np
import matplotlib.pyplot as plt
import random
import math

# --- Configuration ---
MATFILE = "scrambled_lena.mat"
TILES = 4
ORIENTS = 8
SEED = 77
T0 = 50000.0
TMIN = 1e-16
COOL = 0.999994
ITER = 2000000    # as high as you want (max allowed)
EDGE = 6

random.seed(SEED)
np.random.seed(SEED)

def load_ascii_mat_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    dims = data_lines[0].split()
    rows, cols = int(dims[0]), int(dims[1])
    pixel_lines = data_lines[1:]
    if len(pixel_lines) != rows * cols:
        raise ValueError(f"Expected {rows*cols} pixels but got {len(pixel_lines)}")
    pixels = np.array([int(v) for v in pixel_lines], dtype=np.uint8)
    return pixels.reshape(rows, cols)

img = load_ascii_mat_file(MATFILE)
n = img.shape[0]
assert img.shape[0] == img.shape[1], "Image must be square"
assert n % TILES == 0, "Image not divisible by TILES"
tsz = n // TILES

flip0 = lambda x: x
flip1 = lambda x: np.rot90(x, 1)
flip2 = lambda x: np.rot90(x, 2)
flip3 = lambda x: np.rot90(x, 3)
flip4 = lambda x: np.fliplr(x)
flip5 = lambda x: np.flipud(x)
flip6 = lambda x: np.flipud(np.rot90(x, 1))
flip7 = lambda x: np.fliplr(np.rot90(x, 1))
xfm_choices = [flip0, flip1, flip2, flip3, flip4, flip5, flip6, flip7]

pieces = []
for i in range(TILES):
    for j in range(TILES):
        tile = img[i*tsz:(i+1)*tsz, j*tsz:(j+1)*tsz]
        for k in xfm_choices:
            pieces.append(k(tile).copy())

def edge_gradient(arr):
    g1 = np.abs(np.gradient(arr.astype(float), axis=1))
    g2 = np.abs(np.gradient(arr.astype(float), axis=0))
    return g1 + g2
grads = [edge_gradient(tile) for tile in pieces]

def enhanced_mse_score(board, pieces, grads, tile_size, num_blocks, orientations, edge_width=EDGE):
    total_cost = 0
    corner_penalty_wt = 5

    for i in range(num_blocks):
        for j in range(num_blocks - 1):
            idx1, idx2 = board[i, j], board[i, j + 1]
            p1, p2 = pieces[idx1], pieces[idx2]
            g1, g2 = grads[idx1], grads[idx2]
            # Edge difference now as MSE
            total_cost += np.mean((p1[:, -edge_width:].astype(int) - p2[:, :edge_width].astype(int)) ** 2)
            total_cost += 0.35 * np.mean((g1[:, -edge_width:] - g2[:, :edge_width]) ** 2)
            corner_diff = (abs(int(p1[0, -1]) - int(p2[0, 0])) + abs(int(p1[-1, -1]) - int(p2[-1, 0])))
            total_cost += corner_penalty_wt * corner_diff

    for i in range(num_blocks - 1):
        for j in range(num_blocks):
            idx1, idx2 = board[i, j], board[i + 1, j]
            p1, p2 = pieces[idx1], pieces[idx2]
            g1, g2 = grads[idx1], grads[idx2]
            total_cost += np.mean((p1[-edge_width:, :].astype(int) - p2[:edge_width, :].astype(int)) ** 2)
            total_cost += 0.35 * np.mean((g1[-edge_width:, :] - g2[:edge_width, :]) ** 2)
            corner_diff = (abs(int(p1[-1, 0]) - int(p2[0, 0])) + abs(int(p1[-1, -1]) - int(p2[0, -1])))
            total_cost += corner_penalty_wt * corner_diff

    return total_cost

def show_grid(G, out, block_list=pieces, size=tsz):
    out_img = np.zeros((TILES*size, TILES*size), np.uint8)
    for i in range(TILES):
        for j in range(TILES):
            out_img[i*size:(i+1)*size, j*size:(j+1)*size] = block_list[G[i, j]]
    plt.imsave(out, out_img, cmap='gray')
    return out_img

def anneal(start):
    grid = start.copy()
    best = grid.copy()
    min_score = enhanced_mse_score(grid, pieces, grads, tsz, TILES, ORIENTS)
    current_score = min_score
    T = T0
    score_progress = []
    unchanged_count = 0
    intervals = ITER//10  # print every 1/5th of total iterations

    for t in range(1, ITER + 1):
        g2 = grid.copy()
        act = random.choices(['swap', 'rotate', 'remap'], [0.6, 0.3, 0.1])[0]
        if act == 'swap':
            if random.random() < 0.2:
                r1, r2 = np.random.randint(0, TILES, 2)
                g2[[r1, r2], :] = g2[[r2, r1], :]
            elif random.random() < 0.2:
                c1, c2 = np.random.randint(0, TILES, 2)
                g2[:, [c1, c2]] = g2[:, [c2, c1]]
            else:
                a, b = np.random.randint(0, TILES, 2), np.random.randint(0, TILES, 2)
                g2[a[0], a[1]], g2[b[0], b[1]] = g2[b[0], b[1]], g2[a[0], a[1]]
        elif act == 'rotate':
            a = np.random.randint(0, TILES, 2)
            val = g2[a[0], a[1]]
            rot = (val % ORIENTS + np.random.choice([-1, 1])) % ORIENTS
            vbase = val // ORIENTS
            g2[a[0], a[1]] = vbase * ORIENTS + rot
        else:
            a, b = np.random.randint(0, TILES, 2), np.random.randint(0, TILES, 2)
            if tuple(a) != tuple(b):
                v1, v2 = g2[a[0], a[1]], g2[b[0], b[1]]
                v1b, v2b = v1 // ORIENTS, v2 // ORIENTS
                g2[a[0], a[1]] = v2b * ORIENTS + (v1 % ORIENTS)
                g2[b[0], b[1]] = v1b * ORIENTS + (v2 % ORIENTS)
        new_score = enhanced_mse_score(g2, pieces, grads, tsz, TILES, ORIENTS)
        if new_score < current_score or math.exp((current_score - new_score) / T) > random.random():
            grid = g2
            current_score = new_score
            if current_score < min_score:
                best = grid.copy()
                min_score = current_score
        T *= COOL
        # Stopping early if score hasn't improved in 3 intervals
        if t % intervals == 0:
            print(f"Iter {t}/{ITER}, current best score {min_score:.1f}")
            score_progress.append(min_score)
            if len(score_progress) >= 3 and all(score_progress[-1] == score_progress[-i] for i in range(2, 3)):
                print(f"No best score improvement in 3 checks, stopping early at {t}/{ITER}")
                break
    return best, min_score

initial = (np.arange(TILES * TILES) * ORIENTS).reshape(TILES, TILES)
show_grid(initial, 'jigsaw_before.png')
print("Simulated Annealing in progress...")
soln, energy = anneal(initial)
show_grid(soln, 'jigsaw_after.png')
print(f"Annealing done. Final score: {energy:.1f}")
print("Saved: 'jigsaw_before.png', 'jigsaw_after.png'.")
