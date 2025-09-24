from collections import deque

# Swaps rabbit at ri with empty at ei, for label lbl ("Step" or "Leap")
def swap(cfg, ri, ei, lbl):  # ri: rabbit index, ei: empty index
    swaps = []
    rb = cfg[ri]  # rabbit type
    tmp = cfg.copy()
    tmp[ri], tmp[ei] = '_', rb
    desc = f"{lbl} {rb} at {ri} → {ei}"
    swaps.append((tmp, desc))
    return swaps

# Returns all valid moves for rabbit at ri to empty ei
def leap_ops(cfg, ri, ei):  # ri: rabbit idx, ei: empty idx
    ops = []
    k = cfg[ri]
    if k == 'E' and ri + 1 == ei:
        ops += swap(cfg, ri, ei, "Step")
    if k == 'W' and ri - 1 == ei:
        ops += swap(cfg, ri, ei, "Step")
    if k == 'E' and ri + 2 == ei:
        ops += swap(cfg, ri, ei, "Leap")
    if k == 'W' and ri - 2 == ei:
        ops += swap(cfg, ri, ei, "Leap")
    return ops

# Uses DFS to solve the rabbit leap problem (may not be optimal)
def dfs():
    st = ['E', 'E', 'E', '_', 'W', 'W', 'W']  # Start
    gl = ['W', 'W', 'W', '_', 'E', 'E', 'E']  # Goal
    vis = {tuple(st)}
    stk = [(st, [])]

    while stk:
        cfg, path = stk.pop()
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
                        stk.append((nxt, path + [desc]))
    return None, -1

# Main function: runs DFS and displays solution
def main():

    print("Solving the Rabbit Leap Problem using DFS...")

    moves, steps = dfs()


    if moves is None:
        print("No solution found.")
    else:
        print(f"\nSolution found in {steps} steps:")
        for j, m in enumerate(moves, 1):
            print(f"Step {j}: {m}")


if __name__ == "__main__":
    main()
