import numpy as np
import scipy
import matplotlib.pyplot as plt

np.random.seed(70)


n_rows, n_cols = 5, 5

n_flows = 25
total_vehicles = 300
duration = 900
alphas = np.reshape(np.random.rand(n_flows) * 9 + 1, (n_rows, n_cols))
betas = np.reshape(np.random.rand(n_flows) * 9 + 1, (n_rows, n_cols))
probs = np.random.rand(n_flows)
probs = np.reshape(probs / np.sum(probs), (n_rows, n_cols))
vehicles = np.round(probs * total_vehicles).astype(np.int32)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, figsize=(10,7.5))

for r in range(n_rows):
  for c in range(n_cols):
    ax = axs[r, c]
    alpha, beta, n_veh = alphas[r, c], betas[r, c], vehicles[r, c]
    time = np.linspace(0, 1, 100)
    rate = scipy.stats.beta.pdf(time, alpha, beta) * ((n_veh / duration))
    time = time * duration
    departures = scipy.stats.beta.ppf(np.random.rand(n_veh),
                                      alpha, beta) * duration
    ax.plot(time, rate)
    ax.scatter(departures, np.zeros(departures.shape), color="red", marker="|", alpha=0.3, s=300)
    ax.text(0.01, 0.975, f"$n={n_veh}$, $\\alpha={np.round(alpha, 1)}$, $\\beta={np.round(beta, 1)}$", fontsize=9, ha="left", va="top", transform=ax.transAxes)
    ax.set_ylim(top=0.1)
    ax.set_xticks([0, duration])

fig.supxlabel('Time (seconds)')
fig.supylabel('Rate (vehicles/second)')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("random-traffic.pdf")