import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import random
import os.path

# expected usage: many_rand_matrices.py
# this tries the seeds 1-100 for generating random matrices and testing them against this neural network

# Define your neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.step1 = nn.Linear(2, 128, bias=True)
        self.step2 = nn.ReLU()
        self.step3 = nn.Linear(128, 64, bias=True)
        self.step4 = nn.ReLU()
        self.step5 = nn.Linear(64, 16, bias=True)
        self.step6 = nn.ReLU()
        self.step7 = nn.Linear(16, 2, bias=True)

        # multi-class classification adapted from ChatGPT
        self.step8 = nn.Softmax(dim=1)

    def forward(self, x):

        # RUN IT ON A GPU if it exists
        if torch.cuda.is_available():
            x = x.to("cuda")

        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        x = self.step5(x)
        x = self.step6(x)
        x = self.step7(x)
        x = self.step8(x)
        
        return x

def train_model(inputs, desired_outputs, num_epochs=500, learning_rate=0.01):
    # Convert inputs and desired_outputs to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)

    # Create a DataLoader to handle batching (if needed)
    dataset = TensorDataset(inputs, desired_outputs)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)  # Adjust batch_size as needed
    
    # Initialize the model
    model = SimpleModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    inputs = inputs.float()
    desired_outputs = desired_outputs.long()
    
    # Training loop
    for _ in range(num_epochs):
        total_loss = 0.0
        for batch_inputs, batch_desired_outputs in dataloader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_inputs)  # Forward pass

            batch_desired_outputs = batch_desired_outputs.long()

            loss = criterion(outputs, batch_desired_outputs)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model's parameters
            cur_item = loss.item()
            total_loss += cur_item
        
        scheduler.step()    

    # Return the trained model
    return model

def test_model(trained_model, new_data, expected_result):
    test_input = torch.tensor(new_data, dtype=torch.float32)

    test_output = trained_model(test_input)

    predicted_classes = torch.argmax(test_output, dim=1)

    return float(sum(expected_result == predicted_classes)/len(test_input))

accuracy_df = pd.DataFrame({
    'A': [],
    'B': [],
    'detA': [],
    'detB': [],
    'accuracy': []
})

for rand_seed in range(1, 100): 
    print(rand_seed)
    random.seed(rand_seed)

    A = np.array([[random.randint(0, 6) - 3, random.randint(0, 6) - 3], [random.randint(0, 6) - 3, random.randint(0, 6) - 3]])
    B = np.array([[random.randint(0, 6) - 3, random.randint(0, 6) - 3], [random.randint(0, 6) - 3, random.randint(0, 6) - 3]])

    # if we don't have the non-test csv file:
    if not os.path.isfile(f'two_rand_matrices_seed_{rand_seed}.csv'):

        data = {
            'val1': [],
            'val2': [],
            'last_matrix': []
        }

        test_df = pd.DataFrame(data)

        random.seed(rand_seed)
        for _ in range(1, 10000):
            
            x = [str(random.randint(1, 2)) for _ in range(0, 10)]
            all_digits = "".join(x)

            m = np.random.randint(1, 21)

            cur_array = np.array([0, m])
            for i in all_digits:
                if i=="1":
                    cur_array = cur_array @ A
                elif i=="2":
                    cur_array = cur_array @ B

            # adapted from https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe

            df2 = pd.DataFrame([[cur_array[0], cur_array[1], all_digits[-1]]], columns=['val1', 'val2', 'last_matrix'])
            test_df = pd.concat([df2, test_df])

        test_df['last_matrix'] = test_df['last_matrix'].astype(int)

        test_df.to_csv(f'two_rand_matrices_seed_{rand_seed}.csv', index=False)

    # if we don't have the test file: 
    if not os.path.isfile(f'two_rand_matrices_seed_{rand_seed}_test.csv'):
        data = {
            'val1': [],
            'val2': [],
            'last_matrix': []
        }

        test_df = pd.DataFrame(data)

        random.seed(49)
        for _ in range(1, 10000):
            
            x = [str(random.randint(1, 2)) for _ in range(0, 10)]
            all_digits = "".join(x)

            cur_array = np.array([0, 1])
            for i in all_digits:
                if i=="1":
                    cur_array = cur_array @ A
                elif i=="2":
                    cur_array = cur_array @ B

            # adapted from https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe

            df2 = pd.DataFrame([[cur_array[0], cur_array[1], all_digits[-1]]], columns=['val1', 'val2', 'last_matrix'])
            test_df = pd.concat([df2, test_df])

        test_df['last_matrix'] = test_df['last_matrix'].astype(int)

        test_df.to_csv(f'two_rand_matrices_seed_{rand_seed}_test.csv', index=False)


    df = pd.read_csv(f"two_rand_matrices_seed_{rand_seed}.csv")

    just_input = df.drop('last_matrix', axis=1)

    # adapted from https://stackoverflow.com/questions/43898035/pandas-combine-column-values-into-a-list-in-a-new-column
    input_data = np.array(just_input.values.tolist())

    df['last_matrix'] = df['last_matrix'] - 1

    desired_output = torch.tensor(df['last_matrix'].tolist(), dtype=torch.float32).long()

    trained_model = train_model(input_data, desired_output, 100, learning_rate=0.001)

    test_df = pd.read_csv(f"two_rand_matrices_seed_{rand_seed}_test.csv")
    just_input_test = test_df.drop('last_matrix', axis=1)
    input_data_test = np.array(just_input_test.values.tolist())
    test_df['last_matrix'] = test_df['last_matrix'] - 1
    desired_output = torch.tensor(test_df['last_matrix'].tolist(), dtype=torch.float32).long()

    tested_accuracy = test_model(trained_model, input_data_test, desired_output)

    df2 = pd.DataFrame([[A, B, round(np.linalg.det(A)), round(np.linalg.det(B)), tested_accuracy]], columns=['A', 'B', 'detA', 'detB', 'accuracy'])
    accuracy_df = pd.concat([accuracy_df, df2])

accuracy_df.to_csv(f'summarize_rand_matrices_500_epoch.csv', index=False)