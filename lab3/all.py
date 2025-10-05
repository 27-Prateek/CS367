import random
from itertools import combinations

def gen_ksat(k, m, n):
    out = []
    for _ in range(m):
        vs = random.sample(range(1, n+1), k)
        cl = [random.choice([-1, 1]) * x for x in vs]
        out.append(cl)
    return out

def hill_climb(fml, n, h, tries=100, flips=1000):
    st = 0
    for _ in range(tries):
        X = [None] + [random.choice([False, True]) for _ in range(n)]
        for _ in range(flips):
            st += 1
            if h(fml, X) == 0:
                return X, st, True
            curr_h = h(fml, X)
            best_h = curr_h
            best_vars = []
            for j in range(1, n+1):
                X[j] = not X[j]
                new_h = h(fml, X)
                if new_h < best_h:
                    best_h = new_h
                    best_vars = [j]
                elif new_h == best_h:
                    best_vars.append(j)
                X[j] = not X[j]
            if best_h < curr_h:
                j = random.choice(best_vars)
                X[j] = not X[j]
            else:
                break
    return None, st, False

def beam(fml, n, width, h, steps=1000):
    pool = [[None] + [random.choice([False, True]) for _ in range(n)] for _ in range(width)]
    for st in range(steps):
        for X in pool:
            if h(fml, X) == 0:
                return X, st+1, True
        nxt = []
        for X in pool:
            for j in range(1, n+1):
                Y = X.copy()
                Y[j] = not Y[j]
                nxt.append(Y)
        nxt.sort(key=lambda a: h(fml, a))
        pool = nxt[:width]
    return None, steps, False

def vnd(fml, n, h, tmax=10, kmax=3):
    st = 0
    for _ in range(tmax):
        A = [None] + [random.choice([False, True]) for _ in range(n)]
        k = 1
        while k <= kmax:
            st += 1
            if h(fml, A) == 0:
                return A, st, True
            curr_h = h(fml, A)
            best_h = curr_h
            best_flip = None
            for flip_set in combinations(range(1, n+1), k):
                for i in flip_set:
                    A[i] = not A[i]
                new_h = h(fml, A)
                if new_h < best_h:
                    best_h = new_h
                    best_flip = flip_set
                for i in flip_set:
                    A[i] = not A[i]
            if best_h < curr_h and best_flip is not None:
                for i in best_flip:
                    A[i] = not A[i]
                k = 1
            else:
                k += 1
    return None, st, False
