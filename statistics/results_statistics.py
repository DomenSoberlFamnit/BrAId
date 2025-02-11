import pandas as pd

include_raised_axles = True

df = pd.read_csv('hybrid-results.csv')
df = df[['RP2', 'ROAD', 'NN_PREDICTION', 'CAMERA', 'AGREE/POSITIVE', 'IS_RAISED']]

siwim = {}
cv = {}
hybrid = {}
cnt_road = {}
cnt_camera = {}
cnt_raised = {}

cnt_agreed_raised = 0
cnt_all = 0

for index, row in df.iterrows():
    rp2 = str(row['RP2'])
    nn = str(row['NN_PREDICTION'])
    road = str(row['ROAD'])
    camera = str(row['CAMERA'])

    raised_axles = row['IS_RAISED'] == 1

    if not include_raised_axles and raised_axles:
        continue

    positive = row['AGREE/POSITIVE'] == 1

    # SIWIM
    if rp2 not in siwim:
        siwim[rp2] = {'tp': 0, 'fp': 0, 'fn': 0}
    if road not in siwim:
        siwim[road] = {'tp': 0, 'fp': 0, 'fn': 0}

    if rp2 == road:
        siwim[rp2]['tp'] += 1
    else:
        siwim[road]['fn'] += 1
        siwim[rp2]['fp'] += 1

    # CV
    if nn not in cv:
        cv[nn] = {'tp': 0, 'fp': 0, 'fn': 0}
    if camera not in cv:
        cv[camera] = {'tp': 0, 'fp': 0, 'fn': 0}

    if nn == camera:
        cv[nn]['tp'] += 1
    else:
        cv[camera]['fn'] += 1
        cv[nn]['fp'] += 1

    # Hybrid
    if rp2 not in hybrid:
        hybrid[rp2] = {'tp': 0, 'fp': 0, 'fn': 0}
    if road not in hybrid:
        hybrid[road] = {'tp': 0, 'fp': 0, 'fn': 0}

    if positive:
        if rp2 == road:
            hybrid[rp2]['tp'] += 1
        else:
            hybrid[rp2]['fp'] += 1
    else:
        if rp2 == road:
            hybrid[rp2]['fn'] += 1

    # count
    if road not in cnt_road:
        cnt_road[road] = 0
    cnt_road[road] += 1

    if camera not in cnt_camera:
        cnt_camera[camera] = 0
    cnt_camera[camera] += 1

    if road not in cnt_raised:
        cnt_raised[road] = 0

    if raised_axles:
        cnt_raised[road] += 1

    if positive and raised_axles:
        cnt_agreed_raised += 1

    cnt_all += 1

n = 0
sum_siwim_p = 0
sum_siwim_r = 0
sum_cv_p = 0
sum_cv_r = 0
sum_hybrid_p = 0
sum_hybrid_r = 0
for groups in siwim:
    tp = siwim[groups]['tp']
    fp = siwim[groups]['fp']
    fn = siwim[groups]['fn']

    if tp > 0:
        siwim_p = tp / (tp + fp)
        siwim_r = tp / (tp + fn)
        siwim_ca = tp / cnt_road[groups]
    else:
        continue

    if groups not in cv:
        continue

    tp = cv[groups]['tp']
    fp = cv[groups]['fp']
    fn = cv[groups]['fn']

    if tp > 0:
        cv_p = tp / (tp + fp)
        cv_r = tp / (tp + fn)
        cv_ca = tp / cnt_camera[groups]
    else:
        continue

    if groups not in hybrid:
        continue

    tp = hybrid[groups]['tp']
    fp = hybrid[groups]['fp']
    fn = hybrid[groups]['fn']

    if tp > 0:
        hybrid_p = tp / (tp + fp)
        hybrid_r = tp / (tp + fn)
        hybrid_ca = tp / cnt_road[groups]
    else:
        continue

    sum_siwim_p += siwim_p * (cnt_road[groups] / cnt_all)
    sum_siwim_r += siwim_r * (cnt_road[groups] / cnt_all)
    sum_cv_p += cv_p * (cnt_camera[groups] / cnt_all)
    sum_cv_r += cv_r * (cnt_camera[groups] / cnt_all)
    sum_hybrid_p += hybrid_p * (cnt_road[groups] / cnt_all)
    sum_hybrid_r += hybrid_r * (cnt_road[groups] / cnt_all)

    n += 1
    print(f'{n} {groups} {cnt_road[groups]} {cnt_camera[groups]} {cnt_raised[groups]} {(100 * siwim_ca):.2f} {(100 * siwim_p):.2f} {(100 * siwim_r):.2f} {(100 * cv_ca):.2f} {(100 * cv_p):.2f} {(100 * cv_r):.2f} {(100 * hybrid_ca):.2f} {(100 * hybrid_p):.2f} {(100 * hybrid_r):.2f}')

print('---------------')
print(f'{(100 * sum_siwim_p):.2f} {(100 * sum_siwim_r):.2f} {(100 * sum_cv_p):.2f} {(100 * sum_cv_r):.2f} {(100 * sum_hybrid_p):.4f} {(100 * sum_hybrid_r):.4f}')
print(f'Positive raised axles: {cnt_agreed_raised}')
print(f'All cases: {cnt_all}')
