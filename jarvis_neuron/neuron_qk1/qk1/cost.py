"""QPU cost ESTIMATE for a fidelity-kernel workflow. Estimate only; not a guarantee."""
def estimate(n_train, n_test, shots=1000, repeats=3, t_shot_s=2.5e-4, calib_circuits=20, overhead_s_per_job=2.0, circuits_per_job=300, balance_s=None, reserve_s=90):
    pairs = n_train*(n_train-1)//2 + n_test*n_train
    circuits = pairs*repeats + calib_circuits
    jobs = -(-circuits // circuits_per_job)
    est = jobs*overhead_s_per_job + circuits*shots*t_shot_s
    out = dict(pair_circuits=pairs, circuits=circuits, jobs=jobs, est_seconds=est, assumptions=dict(shots=shots, repeats=repeats, t_shot_s=t_shot_s))
    if balance_s is not None: out["fits_with_reserve"] = est <= balance_s - reserve_s
    return out
if __name__ == "__main__":
    for nt in (50, 200, 560):
        print(nt, estimate(nt, 140, balance_s=345))
