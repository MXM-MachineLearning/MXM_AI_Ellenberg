
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
file_path = '../Data_Generation/Data_files/bfs_heisenberg_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
data.head()

tolerance = 2
cross_section_plane = data[(abs(data['val3']) < 10) & (abs(data['val2']) < 50) &
                           (abs(data['val3'] - data['val1']) <= tolerance)]


# Create a 2D plot for the cross section close to the plane y=x
plt.figure(figsize=(10, 8))
sc = plt.scatter(cross_section_plane['val1'], cross_section_plane['val2'],
                 c=cross_section_plane['last_matrix'], cmap='viridis', marker='o')

# Title and labels
plt.title('2D Cross Section of Points on (or very close to) the Plane y=x')
plt.xlabel('val1 (a,b)')
plt.ylabel('val2 (c)')  # Since val2 is plotted on the z-axis in the 3D plot

# Colorbar to show the gradient scale
plt.colorbar(sc, label='next move made')
plt.grid(True)
plt.show()
