from collections import deque


# Performs swap of rabbit at index ri with empty space at ei, label="Step" or "Leap"
def swap(cfg, ri, ei, lbl):  # ri: rabbit index, ei: empty index
    swaps = []
    rb = cfg[ri]  # rabbit type
    tmp = cfg.copy()
    tmp[ri], tmp[ei] = '_', rb
    desc = f"{lbl} {rb} at {ri} → {ei}"
    swaps.append((tmp, desc))
    return swaps

# Returns all possible valid moves for rabbit at ri to empty at ei
def leap_ops(cfg, ri, ei):  # ri: rabbit index, ei: empty index
    ops = []
    kind = cfg[ri]
    if kind == 'E' and ri + 1 == ei:
        ops += swap(cfg, ri, ei, "Step")
    if kind == 'W' and ri - 1 == ei:
        ops += swap(cfg, ri, ei, "Step")
    if kind == 'E' and ri + 2 == ei:
        ops += swap(cfg, ri, ei, "Leap")
    if kind == 'W' and ri - 2 == ei:
        ops += swap(cfg, ri, ei, "Leap")
    return ops

# Uses BFS to find the shortest solution for the rabbit leap problem
def bfs():
    st = ['E', 'E', 'E', '_', 'W', 'W', 'W']  # Start config
    gl = ['W', 'W', 'W', '_', 'E', 'E', 'E']  # Goal config
    vis = {tuple(st)}
    q = deque([(st, [])])

    while q:
        cfg, path = q.popleft()
        if cfg == gl:
            return path, len(path)
        ei = cfg.index('_')
        for i in range(len(cfg)):
            if cfg[i] != '_':
                moves = leap_ops(cfg, i, ei)
                for nxt, desc in moves:
                    tup = tuple(nxt)
                    if tup not in vis:
                        vis.add(tup)
                        q.append((nxt, path + [desc]))
    return None, -1

# Main function: runs BFS and displays solution
def main():
    print("Solving the Rabbit Leap Problem using BFS...")

    moves, steps = bfs()


    if moves is None:
        print("No solution found.")
    else:
        print(f"\nOptimal Solution Found in {steps} steps:")
        for j, m in enumerate(moves, 1):
            print(f"Step {j}: {m}")

        print(f"BFS always guarantees the shortest solution, so it is optimal.")

if __name__ == "__main__":
    main()
