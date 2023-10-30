import matplotlib
matplotlib.use('TkAgg')  # Set the backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
from PIL import Image
import os
import time

# Load the data
df = pd.read_csv('../Data_Generation/Data_files/Q_moves_scores.csv')
x = df['val1'].values
y = df['val2'].values
z = df['val3'].values
q_values = df['Q_table_val'].values

# Define the ranges
ranges = [
    (9750.21, 10000.00, 'range_9750_21_to_10000'),
    (9500.42, 9750.21, 'range_9500_42_to_9750_21'),
    (None, 0, 'range_below_0'),
    (7502.08, 9500.42, 'range_7502_08_to_9500_42'),
    (1000, 7502.08, 'range_1000_to_7502_08')
]

temp_files = []

# Generate GIFs for each range
for lower, upper, name in ranges:
    # Filter points based on the specified range
    if lower is None:
        mask = (q_values <= upper)
    else:
        mask = (q_values > lower) & (q_values <= upper)

    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    q_filtered = q_values[mask]

    # Create 3D scatter plot and save frames for GIF
    frames = []
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c=q_filtered, cmap='viridis', s=40)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Q_table_val')

    label = f'Q values in range ({lower}, {upper}]'
    ax.set_title(label)

    for angle in range(0, 360, 2):
        ax.view_init(30, angle)
        temp_file = f"temp_{angle}.png"
        plt.savefig(temp_file)
        with Image.open(temp_file) as img:
            frames.append(img.copy())  # copy image to frames list
            img.close()  # explicitly close the image

        try:
            os.remove(temp_file)
        except PermissionError:
            print(f"Could not delete {temp_file}. Skipping.")

    frames[0].save(f'{name}.gif',
                   save_all=True, append_images=frames[1:],
                   optimize=False, duration=40, loop=0)

# Delete all temporary files


