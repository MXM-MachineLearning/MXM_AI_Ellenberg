
import numpy as np
from collections import deque
import pandas as pd

# Define the necessary matrices
A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
B = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
C = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
D = np.array([[1, 0, 0], [0, 1, -1], [0, 0, 1]])
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def matrix_to_tuple(matrix):
    return (matrix[0, 1], matrix[0, 2], matrix[1, 2])

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

# This outputs a dataframe with elements of the heisenberg group and their sets of optimal steps back to the origin
def modified_bfs(start, myDF):
    visited = set()
    queue = deque([[start, 0, set()]])
    
    while queue:
        cur_mat, num_steps, last_matrices = queue.popleft()
        cur_tuple = matrix_to_tuple(cur_mat)

        existing_rows = myDF[(myDF['val1'] == cur_tuple[0]) & (myDF['val2'] == cur_tuple[1]) & (myDF['val3'] == cur_tuple[2])]
        
        if not existing_rows.empty:
            if existing_rows.iloc[0]['num_steps'] == num_steps:
                updated_last_matrices = existing_rows.iloc[0]['last_matrices'].union(last_matrices)
                myDF.at[existing_rows.index[0], 'last_matrices'] = updated_last_matrices
        else:
            new_row = pd.DataFrame([{
                'val1': cur_tuple[0],
                'val2': cur_tuple[1],
                'val3': cur_tuple[2],
                'last_matrices': last_matrices,
                'num_steps': num_steps
            }])
            myDF = pd.concat([new_row, myDF])

        if num_steps >= 10:
            break

        for neighbor in [0, 1, 2, 3]:
            new_mat = apply_mat(cur_mat, neighbor)
            new_tuple = matrix_to_tuple(new_mat)
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_last_matrices = last_matrices.copy()
                new_last_matrices.add(neighbor)
                queue.append([new_mat, num_steps + 1, new_last_matrices])

    return myDF

# Initialize the DataFrame
data = pd.DataFrame({
    'val1': [],
    'val2': [],
    'val3': [],
    'last_matrices': [],
    'num_steps': []
})

# Apply the modified BFS function
data = modified_bfs(I, data)

# Display the DataFrame
print(data.head())

# data.to_csv("../Data_Generation/Data_files/bfs_heisenberg_data_with_sets.csv", index=False)