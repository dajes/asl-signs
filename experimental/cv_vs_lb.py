import numpy as np
from matplotlib import pyplot as plt

with open('cv_vs_lb.csv', 'r') as f:
    lines = f.readlines()
print(lines.pop(0))
models, lbs, cvs = zip(*[[float(e.split()[0]) for e in line.split(',')] for line in lines])

x = np.array(lbs)
y = np.array(cvs)
linspace = np.linspace(x.min(), 1, 1000)

lb_targets = np.arange(0.75, .9, .01)
lb_xs = []

for order in range(1, 3):
    fit = np.polynomial.polynomial.polyfit(x, y, order)
    preds = np.polynomial.polynomial.polyval(linspace, fit).clip(0, 1)
    matching = np.abs(lb_targets[:, None] - preds[None]).argmin(1)
    lb_xs.append(linspace[matching])
    plt.plot(linspace, preds, label=f'Order {order}')
plt.legend()

plt.xlabel('CV')
plt.ylabel('LB')
plt.scatter(x, y, s=200)
plt.show()

lb_xs = np.stack(lb_xs, 1)
print('LB  ', 'Predicted val', sep='\t')
for lb_target, lb_x in zip(lb_targets, lb_xs):
    print(f'{lb_target:.2f}', f'{lb_x.mean():.4f} ± {max(np.abs(lb_x.mean() -lb_x)):.3f}', sep='\t')

print()
