import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation

# Creating 3D figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

df = pd.read_csv("../Data_Generation/Data_files/bfs_heisenberg_data.csv")

# this filtering is necessary to limit the number of points we plot because otherwise this is slow
df = df[df['num_steps'] < 10]

# Create data
x = df["val1"]
y = df['val3']
z = df['val2']

first_move_A = df[df['last_matrix'] == 0]
first_move_B = df[df['last_matrix'] == 1]
first_move_C = df[df['last_matrix'] == 2]
first_move_D = df[df['last_matrix'] == 3]

def plot_data():
    ax.scatter(first_move_A["val1"], first_move_A["val2"], first_move_A["val3"], color="red")
    ax.scatter(first_move_B["val1"], first_move_B["val2"], first_move_B["val3"], color="green")
    ax.scatter(first_move_C["val1"], first_move_C["val2"], first_move_C["val3"], color="black")
    ax.scatter(first_move_D["val1"], first_move_D["val2"], first_move_D["val3"], color="blue")

# Define update function for animation
def update(angle):
    ax.cla()  # Clear axis
    plot_data()  # Plot data
    ax.view_init(45, angle)  # Update view angle

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)

# Save animation as GIF
ani.save('rotation.gif', writer='imagemagick', fps=20)

plt.show()
