import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('paper_style.mplstyle')

sigma_high_2 = .5
sigma_low_2 = .1


def gaussian(x, sigma, mu):
    return (sigma * np.sqrt(2 * np.pi))**(-1) * np.exp(-(1/2) * ((x - mu)/sigma)**2)

fig, ax = plt.subplots(figsize = (2, 1.5))

y = np.linspace(-1.5,1.5, 100)
x_reliable = gaussian(y, np.sqrt(sigma_low_2), 0)
x_noisy = gaussian(y, np.sqrt(sigma_high_2), 0)

ax.plot(y, x_reliable)
ax.plot(y, x_noisy)
ax.set_xlabel("frequency")
ax.set_ylabel("response")

print(sum(x_reliable))
print(sum(x_noisy))

sns.despine()
plt.savefig("../results/plots/response_hist.pdf")
plt.show()