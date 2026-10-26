import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

p = np.array([0.5, 0.5])
q = np.array([0.8, 0.2])

H = entropy(p, base=2)
CE = -np.sum(p * np.log2(q))
KL = entropy(p, q, base=2)

print("Entropy:", round(H, 4))
print("Cross-Entropy:", round(CE, 4))
print("KL Divergence:", round(KL, 4))

labels = ["Entropy", "Cross-Entropy", "KL Divergence"]
values = [H, CE, KL]

plt.bar(labels, values)
plt.ylabel("Value (bits)")
plt.title("Information Theory Measures")
plt.show()
