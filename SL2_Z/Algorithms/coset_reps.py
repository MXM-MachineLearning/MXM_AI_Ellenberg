import itertools
import numpy as np
import pandas as pd

poss = [0, 1]
gamma_cosets = []

for e in itertools.product(poss, poss, poss, poss):
    a = np.array([[e[0],e[1]],[e[2],e[3]]])
    if np.linalg.det(a) % 2 == 1:
        gamma_cosets.append(a)

cosets = gamma_cosets + [A @ np.array([[-1, 0],[0, -1]]) for A in gamma_cosets]

print(gamma_cosets)
[print(A) for A in cosets]
print(len(cosets))