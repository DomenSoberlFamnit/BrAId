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

def precision_nn(samples):
    pass_idx = np.where(samples[:,0])[0]
    cnt_tp = np.sum(samples[pass_idx][:,1])
    return 100.0 * cnt_tp / len(pass_idx)

def paired_bootstrap_test(data1, data2, n_boot=5000):
    n = len(data1)
    diffs = np.empty(n_boot)
    
    for i in range(n_boot):
        # sample with replacement
        idx = rng.integers(0, n, n)
        samples1 = data1[idx]
        samples2 = data2[idx]
        mA = precision_nn(samples1)
        mB = precision_nn(samples2)
        diffs[i] = mA - mB  # improvement in metric

    # observed difference
    obs_diff = precision_nn(data1) - precision_nn(data2)
    # 95% confidence interval
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: how often difference ≤ 0 (if positive improvement)
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    
    return obs_diff, (ci_low, ci_high), p_value

filename1= 'stat-vgg19.csv'
filename2 = 'stat-mobilenet.csv'

print(filename1, filename2)

data1 = read_data(filename1)
data2 = read_data(filename2)

rng = np.random.default_rng(0)

obs_diff, ci, pval = paired_bootstrap_test(data1, data2)

print(f"Observed precission difference = {obs_diff:.4f}")
print(f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"Bootstrap p ≈ {pval}")