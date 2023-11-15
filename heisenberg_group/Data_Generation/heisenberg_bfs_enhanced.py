
import numpy as np
from collections import deque
import pandas as pd
import time


start_time = time.time()

DISTANCE = 100
# Define the matrices A, B, C, D, and I used in the Heisenberg group
# A and B are inverses, as are C and D. I is the identity matrix.
A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
B = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
C = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
D = np.array([[1, 0, 0], [0, 1, -1], [0, 0, 1]])
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Convert a matrix to a tuple representation
def matrix_to_tuple(matrix):
    return (matrix[0, 1], matrix[0, 2], matrix[1, 2])

# Apply a specified matrix (A, B, C, or D) to a given matrix
def apply_mat(mat, index):
    if index == 0:
        return mat @ A
    elif index == 1:
        return mat @ B
    elif index == 2:
        return mat @ C
    elif index == 3:
        return mat @ D
    raise ValueError("Invalid index")

# Perform a modified breadth-first search (BFS) in the Heisenberg group
def modified_bfs(start, myDF):
    # Initialize a set to track visited states and a queue for BFS
    visited = set()
    queue = deque([[start, 0, set()]])

    counter = 0
    while queue:

        # Ensure code is running
        if counter % 120 ==0: print(' Counter: ', counter)
        else: print('#', end="")
        counter += 1


        # Pop the next matrix, the number of steps, and the set of last matrices from the queue
        cur_mat, num_steps, last_matrices = queue.popleft()
        cur_tuple = matrix_to_tuple(cur_mat)

        # Check if this state already exists in the DataFrame
        existing_rows = myDF[(myDF['val1'] == cur_tuple[0]) & (myDF['val2'] == cur_tuple[1]) & (myDF['val3'] == cur_tuple[2])]
        
        # If the state exists, update the last_matrices set if the number of steps is the same
        if not existing_rows.empty:
            if existing_rows.iloc[0]['num_steps'] == num_steps:
                updated_last_matrices = existing_rows.iloc[0]['last_matrices'].union(last_matrices)
                myDF.at[existing_rows.index[0], 'last_matrices'] = updated_last_matrices
        else:
            # Add the new state to the DataFrame
            new_row = pd.DataFrame([{
                'val1': cur_tuple[0],
                'val2': cur_tuple[1],
                'val3': cur_tuple[2],
                'last_matrices': last_matrices,
                'num_steps': num_steps
            }])
            myDF = pd.concat([new_row, myDF])

        # Terminate the loop after a fixed number of steps to prevent infinite search
        if num_steps >= DISTANCE:
            break

        # Add new states to the queue by applying matrices A, B, C, and D
        for neighbor in [0, 1, 2, 3]:
            new_mat = apply_mat(cur_mat, neighbor)
            new_tuple = matrix_to_tuple(new_mat)
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_last_matrices = last_matrices.copy()
                new_last_matrices.add(neighbor)
                queue.append([new_mat, num_steps + 1, new_last_matrices])

    return myDF

# Initialize a DataFrame to store the results of the BFS
data = pd.DataFrame({
    'val1': [],
    'val2': [],
    'val3': [],
    'last_matrices': [],
    'num_steps': []
})

# Apply the modified BFS function to find all optimal paths in the Heisenberg group
data = modified_bfs(I, data)
data.to_csv("../Data_Generation/Data_files/bfs_heisenberg_data_with_sets.csv", index=False)
# Print the first few rows of the DataFrame to display the results
print(data.head())
print('Finished')
print("--- %s seconds ---" % (time.time() - start_time))