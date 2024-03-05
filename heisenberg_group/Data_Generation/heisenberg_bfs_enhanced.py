import numpy as np
from collections import deque
import pandas as pd
import time

start_time = time.time()

DISTANCE = 30
# Matrix definitions (A, B, C, D, I)
A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
B = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
C = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
D = np.array([[1, 0, 0], [0, 1, -1], [0, 0, 1]])
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Helper functions
def matrix_to_tuple(matrix):
    return (matrix[0, 1], matrix[0, 2], matrix[1, 2])

def apply_mat(mat, index):
    return [A, B, C, D][index] @ mat

# This BFS saves both optimal sets and all optimal paths.
def modified_bfs(start, distance):
    queue = deque([[start, 0, [[]]]])
    state_distance = {}
    optimal_paths = {}
    last_steps = {}

    while queue:
        cur_mat, num_steps, paths = queue.popleft()
        cur_tuple = matrix_to_tuple(cur_mat)

        if cur_tuple in state_distance and state_distance[cur_tuple] <= num_steps:
            if state_distance[cur_tuple] < num_steps:
                continue
            else:
                # Update existing paths and last steps
                optimal_paths[cur_tuple].extend(paths)
                last_steps[cur_tuple].update([path[-1] for path in paths if path])
        else:
            # Store new paths and last steps
            state_distance[cur_tuple] = num_steps
            optimal_paths[cur_tuple] = paths
            last_steps[cur_tuple] = set([path[-1] for path in paths if path])

        if num_steps >= distance:
            continue

        for neighbor in range(4):
            new_mat = apply_mat(cur_mat, neighbor)
            new_paths = [path + [neighbor] for path in paths]
            queue.append([new_mat, num_steps + 1, new_paths])

    # Prepare data for DataFrame
    new_rows = [{'val1': key[0], 'val2': key[1], 'val3': key[2],
                 'paths': optimal_paths[key], 'last_steps': last_steps[key],
                 'num_steps': state_distance[key]} for key in optimal_paths]

    return pd.DataFrame(new_rows)

# Perform the BFS and save results
data = modified_bfs(I, DISTANCE)
data.to_csv("../Data_Generation/Data_files/bfs_heisenberg_data_with_optimal_paths_and_last_steps.csv", index=False)

# Output
print(data.head())
print('Finished')
print("--- %s seconds ---" % (time.time() - start_time))
