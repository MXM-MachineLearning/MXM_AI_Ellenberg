import matplotlib
matplotlib.use('TkAgg')  # Set the backend
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

# Load the data
df = pd.read_csv('../Data_Generation/Data_files/Q_moves_scores.csv')
x = df['val1'].values
y = df['val2'].values
z = df['val3'].values
q_values = df['Q_table_val'].values

# Define the ranges

temp_files = []

lower = 7502.08
upper = 9500.42
name = 'range_7502_08_to_9500_42'

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
    print(angle)
    ax.view_init(30, angle)
    temp_file = f"./temp_{angle}.png"
    plt.savefig(temp_file)
    print("Hi")
    plt.close()
    with Image.open(temp_file) as img:
        print("dont get here")
        frames.append(img.copy())  # copy image to frames list
        # img.close()  # explicitly close the image

    print("What?")
    try:
        os.remove(temp_file)
    except PermissionError:
        print(f"Could not delete {temp_file}. Skipping.")

frames[0].save(f'{name}.gif',
                save_all=True, append_images=frames[1:],
                optimize=False, duration=40, loop=0)

# Delete all temporary files


