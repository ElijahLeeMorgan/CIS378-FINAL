import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Linear, ReLU, Softmax, Sequential
import torch.optim as optim

'''
As per Bobeldyk's suggestion, we can use supervised learning to train our model, and use reinforment learning to fine-tune it.
'''
# AI Assisted Code
class SimpleNN(Module):
    def __init__(self, input_size:int=26, output_size:int=4):
        super(SimpleNN, self).__init__()
        self.model = Sequential(
            Linear(input_size, 64),  # Input layer with 26 features
            ReLU(),
            Linear(128, 128),  # Hidden layer
            ReLU(),
            Linear(128, 64),  # Hidden layer
            ReLU(),
            Linear(64, output_size, bias=True),  # Output layer with actionSize neurons (4 actions)
            Softmax(dim=-1)  # Softmax to normalize action probabilities.
            )

    def forward(self, x:torch.Tensor):
        return self.model(x)

# Example Data
#game_states = [[1.0, 50.0, 100.0, 0.0, 0.0, 30.0, ...], [1.0, 60.0, 90.0, 0.0, 0.0, 29.0, ...]]
#actions = [0.0, 2.0]  # Corresponding actions
class DataLoader:
    def __init__(self):
        pass

    def _csvToNumpy(csv_file:str) -> tuple[np.matrix[float], np.vector[float]]:
        # Load CSV data
        # gameData-26floats, action-1float. In order by row.
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1) #TODO Check if the CSV has a header. Porbably not.
        # Split into game states and actions
        game_states = data[:, :-1]  # game states (all columns except the last)
        actions = data[:, -1]  # output actions (last column)
        return game_states, actions

    def _preprocess_data(game_states:np.matrix[float], actions:np.vector[float]):
        # Normalize game states (if needed), convert to tensors.
        game_states = np.array(game_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)  # Actions as integers
        return torch.tensor(game_states), torch.tensor(actions)
    
    def getdata(self, csv_file:str):
        # Load data
        game_states, actions = self._csvToNumpy(csv_file)
        # Preprocess data
        game_states, actions = self._preprocess_data(game_states, actions)
        return game_states, actions

class Trainer:
    def __init__(self, dataPath:str="../trainingData/gameData.csv", model:SimpleNN=None) -> None:
        # Initialize model
        if model is None:
            self.model = SimpleNN(input_size=self._game_states.shape[1], output_size=4)
        else:
            self.model = model

        # Set class _dataLoader object
        self._dataLoader = DataLoader()
        self._game_states = None
        self._actions = None

        self.numEpochs = len(self._actions) # Number of recorded game states.

        # Define loss and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def loadTrainingData(self, dataPath:str="../trainingData/gameData.csv"):
        self._game_states, self._actions = self._dataLoader.getdata(dataPath)

    def train(self):
        # Load data

        
        # Training loop
        epochs = self.numEpochs # Number of CSV rows, aka number of recorded game states.

        for i in range(epochs):

            self.optimizer.zero_grad()
            
            predictions = self.model(self._game_states)  # Forward pass
            
            loss = self.loss_function(predictions, self._actions)  # Compute loss
            loss.backward()  # Backpropagation
            
            self.optimizer.step()  # Update weights
            print(f"Epoch {i + 1}, Loss: {loss.item()}")
        
        print("Training complete.")
        # Save the model
        self.save_model()
        print("Model saved.")
        
    def save_model(self, path:str="supervisedModel.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path:str="supervisedModel.pth"):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
