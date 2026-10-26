import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = 10
p = 0.5

x = np.arange(0, 11)
prob = binom.pmf(x, n, p)

print("Probability of getting 5 heads:",
      binom.pmf(5, n, p))

# Bayes' Theorem
P_A = 0.01
P_B_given_A = 0.90
P_B_given_notA = 0.05

P_B = P_B_given_A * P_A + P_B_given_notA * (1 - P_A)
P_A_given_B = (P_B_given_A * P_A) / P_B

print("Bayes' Theorem Result:", round(P_A_given_B, 4))

plt.bar(x, prob)
plt.xlabel("Number of Heads")
plt.ylabel("Probability")
plt.title("Binomial Distribution")
plt.show()
