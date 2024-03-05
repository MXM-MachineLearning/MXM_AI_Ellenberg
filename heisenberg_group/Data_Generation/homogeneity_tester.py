import pandas as pd
import math
from functools import reduce
import time

start_time = time.time()
FILENAME = 'Data_files/bfs_heisenberg_data_with_sets.csv'
df = pd.read_csv(FILENAME)


# this function makes sure that the given point isn't a multiple of a previously examined point
def special_coprime(a, b, c):
    if a == b == c == 0: return False
    # makes sure the inputs are integers
    a,b,c = int(a), int(b), int(c)

    # Filter out zero elements
    nums = [num for num in [a, b, c] if num != 0]

    # If there's only one nonzero element, check if it's 1
    if len(nums) == 1:
        return nums[0] == 1

    # Calculate the GCD of the remaining elements
    gcd_of_nums = reduce(math.gcd, nums)

    # Return True if the GCD is 1 (i.e., they do not all share a common factor > 1)
    return gcd_of_nums == 1


def find_differences_in_sets(data):
    # Ensure 'last_matrices' is in the correct format (set)
    data['last_matrices'] = data['last_matrices'].apply(lambda x: set(eval(x)))

    # Define a range for the multiplication factor based on your dataset
    max_factor = 10  # This can be adjusted
    counter = 1

    # Iterate through each row in the dataset as a potential "basis"
    for index, basis_row in data.iterrows():
        # Extract basis values and skip if it's (0,0,0)
        basis_vals = (basis_row['val1'], basis_row['val2'], basis_row['val3'])
        if all(v == 0 for v in basis_vals):
            continue

        # Initialize a dictionary to collect sets by factor
        factors_sets = {}

        for factor in range(1, max_factor + 1):
            # Calculate multiples
            multiples = tuple(basis_val * factor if basis_val != 0 else 0 for basis_val in basis_vals)

            # Find matching rows for the generated multiples
            matching_rows = data[
                (data['val1'] == multiples[0]) &
                (data['val2'] == multiples[1]) &
                (data['val3'] == multiples[2])
                ]

            # If no matching rows, continue to next factor
            if matching_rows.empty:
                continue

            # Collect 'last_matrices' sets for matching rows
            for _, matching_row in matching_rows.iterrows():
                factors_sets[factor] = matching_row['last_matrices']

        # Check if the intersection of all collected sets is empty
        if factors_sets:
            all_sets = factors_sets.values()
            intersection = set.intersection(*all_sets)
            if not intersection:
                print(f"Basis: {basis_vals}, Factors' Sets have an empty intersection.")
                print(f"Factors considered: {list(factors_sets.keys())}")
                print("------------------------------------------------")
            if counter % 10000 == 0:
                print('operations', counter)
                runtime = time.time() - start_time
                print('Runtime:', runtime)
                print('operations per second:', counter/runtime)
        counter += 1


find_differences_in_sets(df)

'''
# Loop over all triples in the dataset
for i, row_i in df.iterrows():
    for j, row_j in df.iterrows():
        if i >= j:  # Avoid repeating pairs and self-comparison
            continue
        for k, row_k in df.iterrows():
            if j >= k:  # Avoid repeating pairs and self-comparison
                continue
            # Extract values
            val1_i, val2_i, val3_i = row_i['val1'], row_i['val2'], row_i['val3']
            val1_j, val2_j, val3_j = row_j['val1'], row_j['val2'], row_j['val3']
            val1_k, val2_k, val3_k = row_k['val1'], row_k['val2'], row_k['val3']

            # Check if any two of these three numbers are coprime
            if special_coprime(val1_i, val1_j, val1_k) and special_coprime(val2_i, val2_j, val2_k) and special_coprime(
                    val3_i, val3_j, val3_k):
                # Run your function for the coprime triple
                find_differences_in_sets(val1_i, val2_i, val3_i, df)
                find_differences_in_sets(val1_j, val2_j, val3_j, df)
                find_differences_in_sets(val1_k, val2_k, val3_k, df)
'''