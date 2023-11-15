import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a DataFrame
file_path = '../Data_Generation/Data_files/bfs_heisenberg_data_with_sets.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
print(data.head())

tolerance = 2
cross_section_plane = data[(abs(data['val3']) < 10) & (abs(data['val2']) < 50) &
                           (abs(data['val3'] - data['val1']) <= tolerance)]

# Filter the data for each last matrix applied and create plots
for matrix_index in range(4):
    # Check if the matrix index is in the set of last matrices
    filtered_data = cross_section_plane[cross_section_plane['last_matrices'].str.contains(str(matrix_index))]

    # Create a 2D plot for each subset
    plt.figure(figsize=(10, 8))
    plt.scatter(filtered_data['val1'], filtered_data['val2'], marker='o')

    # Title and labels
    plt.title(f'2D Cross Section of Points with Last Matrix {matrix_index}')
    plt.xlabel('val1 (a,b)')
    plt.ylabel('val2 (c)')

    plt.grid(True)
    plt.show()
