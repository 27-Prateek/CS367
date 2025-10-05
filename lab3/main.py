from all import gen_ksat, hill_climb, beam, vnd

def get_lit_val(l, assign):
    idx = abs(l)
    val = assign[idx]
    if l < 0:
        val = not val
    return val

def h1(fml, assign):
    cnt = 0
    for clause in fml:
        if not any(get_lit_val(l, assign) for l in clause):
            cnt += 1
    return cnt

def h2(fml, assign):
    c = 0
    for clause in fml:
        c += sum(get_lit_val(l, assign) for l in clause)
    return -c

def write_part_b_log(filename, k, m, n, count=3):
    with open(filename, 'w') as f:
        for _ in range(count):
            inst = gen_ksat(k, m, n)
            f.write(f"Random {k}-SAT instance with m={m}, n={n}:\n{inst}\n\n")

def write_part_c_log(filename, n_list, inst_count=3):
    with open(filename, 'w') as f:
        for n in n_list:
            m_list = [int(2*n), int(4*n), int(6*n)]
            for m in m_list:
                f.write(f"\nFor n={n}, m={m}\n")
                for i in range(inst_count):
                    fml = gen_ksat(3, m, n)
                    for h, hn in [(h1, "h1"), (h2, "h2")]:
                        _, st, ok = hill_climb(fml, n, h)
                        N = st * n
                        p = st / N if N > 0 else 0
                        f.write(f"Instance {i}, {hn}, Hill-Climbing: solved={ok}, steps={st}, penetrance={p:.4f}\n")
                        _, st, ok = beam(fml, n, 3, h)
                        N = st * 3 * n
                        p = st / N if N > 0 else 0
                        f.write(f"Instance {i}, {hn}, Beam-3: solved={ok}, steps={st}, penetrance={p:.4f}\n")
                        _, st, ok = beam(fml, n, 4, h)
                        N = st * 4 * n
                        p = st / N if N > 0 else 0
                        f.write(f"Instance {i}, {hn}, Beam-4: solved={ok}, steps={st}, penetrance={p:.4f}\n")
                        _, st, ok = vnd(fml, n, h)
                        N = st * (n**3 / 18)
                        p = st / N if N > 0 else 0
                        f.write(f"Instance {i}, {hn}, VND: solved={ok}, steps={st}, penetrance={p:.4f}\n")

if __name__ == "__main__":
    # Part B output (random k-SAT generator)
    write_part_b_log('b_output.log', k=3, m=8, n=6, count=5)

    # Part C output (solvers and comparison)
    nL = [10, 20]
    write_part_c_log('c_output.log', nL, inst_count=2)
