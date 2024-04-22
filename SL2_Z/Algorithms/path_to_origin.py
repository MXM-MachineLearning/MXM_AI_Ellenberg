import numpy as np
import pandas as pd
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from util import *

inf = float('inf')
U = k_sl2z_2s_gen[0].numpy()
T = k_sl2z_2s_gen[1].numpy()
U_INV = np.linalg.inv(U)
T_INV = np.linalg.inv(T)

def path_to_origin(v: np.array):
    out = v
    repeat = 0
    while np.linalg.norm(out, inf) > 1:
        if (abs(out[0]) > abs(out[1])) and out[0] * out[1] > 0:
            out = U_INV @ out
        elif (abs(out[0]) > abs(out[1])) and out[0] * out[1] < 0:
            out = U @ out 
        elif (abs(out[0]) < abs(out[1])) and out[0] * out[1] > 0:
            out = T_INV @ out 
        elif (abs(out[0]) < abs(out[1])) and out[0] * out[1] < 0:
            out = T @ out
        repeat += 1

        if repeat > 1_000:
            return []
    return out

df = pd.read_csv("./Data_files/sl2_Z.csv")
seen = set()

for index, row in df.iterrows():
    seen.add(tuple(path_to_origin(np.array([row["val1"], row["val3"]]))))

print(seen)