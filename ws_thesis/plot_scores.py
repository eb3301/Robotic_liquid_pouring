import yaml
import numpy as np
import matplotlib.pyplot as plt

# load blocks
iters = []
with open("/tmp/scores_history.yaml") as f:
    for block in yaml.safe_load_all(f):
        iters.append(block["scores"])

# scatter points
xs = []
ys = []
for it, scores in enumerate(iters):
    for s in scores:
        xs.append(it)
        ys.append(s)
plt.scatter(xs, ys)

# trend = media per iter
trend_y = [np.mean(scores) for scores in iters]
plt.plot(range(len(trend_y)), trend_y, color='red', linewidth=2)
plt.plot(range(len(trend_y)),np.ones(len(trend_y)), color='green', linewidth=2)

plt.xlabel("iterazione")
plt.ylabel("score")
plt.show()
