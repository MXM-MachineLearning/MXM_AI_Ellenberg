import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation

# adapted from https://www.geeksforgeeks.org/how-to-animate-3d-graph-using-matplotlib/?ref=lbp

# Creating 3D figure
fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')

df = pd.read_csv("../Data_Generation/Data_files/bfs_heisenberg_data.csv")

# this filtering is necessary to limit the number of points we plot because otherwise this is slow
df = df[df['num_steps']<10]

# Create data
x = df["val1"]
y = df['val3']
z = df['val2']

first_move_A = df[df['last_matrix']==0]
first_move_B = df[df['last_matrix']==1]
first_move_C = df[df['last_matrix']==2]
first_move_D = df[df['last_matrix']==3]

ax.scatter(first_move_A["val1"], first_move_A["val2"], first_move_A["val3"], color="red")
ax.scatter(first_move_B["val1"], first_move_B["val2"], first_move_B["val3"], color="green")
ax.scatter(first_move_C["val1"], first_move_C["val2"], first_move_C["val3"], color="black")
ax.scatter(first_move_D["val1"], first_move_D["val2"], first_move_D["val3"], color="blue")


# 360 Degree view
for angle in range(0, 180):

    ax.view_init(45, angle*2)
    plt.draw()
    plt.pause(.001)
 
plt.show()
