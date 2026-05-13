import numpy as np

def read_data(filename):
    f = open(filename, 'r')

    data = []

    for line in f:
        cols = line.strip().split(',')
        siwim = cols[1]
        nn = cols[8]
        camera = cols[2]
        road = cols[3]
        raised = cols[5]
        agree = cols[10]
        correct = cols[6]

        data.append([True if agree == '1' else False, True if correct == '1' else False])

    f.close()
    return np.array(data)

def precision_siwim(samples):
    cnt_tp = np.sum(samples[:,1])
    return 100.0 * cnt_tp / len(samples)

def precision_nn(samples):
    pass_idx = np.where(samples[:,0])[0]
    cnt_tp = np.sum(samples[pass_idx][:,1])
    return 100.0 * cnt_tp / len(pass_idx)

def paired_bootstrap_test(data, n_boot=5000):
    n = len(data)
    diffs = np.empty(n_boot)
    
    for i in range(n_boot):
        # sample with replacement
        idx = rng.integers(0, n, n)
        samples = data[idx]
        mA = precision_nn(samples)
        mB = precision_siwim(samples)
        diffs[i] = mA - mB  # improvement in metric

    # observed difference
    obs_diff = precision_nn(data) - precision_siwim(data)
    # 95% confidence interval
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: how often difference ≤ 0 (if positive improvement)
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    
    return obs_diff, (ci_low, ci_high), p_value

filename = 'stat-mobilenet.csv'
print(filename)

data = read_data(filename)
rng = np.random.default_rng(0)

obs_diff, ci, pval = paired_bootstrap_test(data)

print(f"Observed precission difference = {obs_diff:.4f}")
print(f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"Bootstrap p ≈ {pval}")