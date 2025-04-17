import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Linear, ReLU, Softmax, Sequential
import torch.optim as optim

'''
Notes:

As per Bobeldyk's suggestion, we can use supervised learning to train our model, and use reinforment learning to fine-tune it.
This will give our reinforcement learning model a good starting point, and save training time.

The model will be a simple feedforward neural network, and subject to change. This doesn't matter as long as there's 26 input and 4 output neurons.
In addion, there should be a softmax layer at the end to normalize the output.
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
            Linear(64, output_size),  # Output layer with actionSize neurons (4 actions)
            Softmax(dim=-1)  # Softmax to normalize action probabilities.
            )

    def feedForward(self, x:torch.Tensor) -> torch.Tensor:
        # Forward pass through the network. Take the max output for predicted action (by index).
        return self.model(x)

# Example Data
#game_states = [[1.0, 50.0, 100.0, 0.0, 0.0, 30.0, ...], [1.0, 60.0, 90.0, 0.0, 0.0, 29.0, ...]]
#actions = [0.0, 2.0]  # Corresponding actions
class DataLoader:
    def __init__(self):
        pass

    def _csvToNumpy(self, csv_file:str, removeEmptyCollumn:bool=False) -> tuple[np.matrix[float], np.matrix[float]]:
        # Load CSV data
        # gameData-26floats, action-1float. In order by row.
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        # Split into game states and actions
        if removeEmptyCollumn:
            data = data[:, np.sum(data, axis=0) != 0]
            print("Empty collumns removed new shape RxC: ", data.shape)
        game_states = data[:, :-1]  # game states (all columns except the last)
        actions = data[:, -1]  # output actions (last column)
        return game_states, actions

    def _preprocess_data(self, game_states:torch.tensor, actions:torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        # Normalize game states (if needed), convert to tensors.
        game_states = np.array(game_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)  # Actions as floats
        return torch.tensor(game_states), torch.tensor(actions)
    
    def getdata(self, csv_file:str, removeEmptyCollumn:bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        # Load data
        game_states, actions = self._csvToNumpy(csv_file, removeEmptyCollumn)
        # Preprocess data
        game_states, actions = self._preprocess_data(game_states, actions)
        return game_states, actions

class Trainer:
    def __init__(self, dataPath:str="../dataEngineering/100Attempts-corruptedSickles.csv", model:SimpleNN=None) -> None:
        self.dataPath = dataPath
        self._game_states = None
        self._actions = None
        self.numEpochs = 0  # Number of epochs for training

        # Set class _dataLoader object
        self._dataLoader = DataLoader()

        # Initialize model
        if model is None:
            self.model = SimpleNN(input_size=len(self._game_states), output_size=4) # Typically 26, but I messed up the sickleXY data while recording. May re-record if nesscessary.
        else:
            self.model = model


        # Define loss and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def loadTrainingData(self, removeEmptyCollumn:bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        self._game_states, self._actions = self._dataLoader.getdata(self.dataPath, removeEmptyCollumn)
        if self._game_states is None or self._actions is None:
            raise ValueError("Game states and actions must be loaded before training.")

    def train(self):
        # Load data

        
        # Training loop
        self.numEpochs = len(self._actions) # Number of recorded game states.
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


def main():
    # Example usage, set args for real use.
    smallModel = SimpleNN(input_size=6, output_size=4)  # Example model initialization
    trainer = Trainer(dataPath="../dataEngineering/100Attempts-corruptedSickles.csv", model=smallModel)  # Example data path
    trainer.loadTrainingData(removeEmptyCollumn=True)
    #trainer.train()

    # Load the model for inference
    #trainer.load_model("supervisedModel.pth")

    # Example inference
    #test_input = torch.randn(1, 6)  # Example input
    #output = trainer.model(test_input)
    #print("Predicted action probabilities:", output)

if __name__ == "__main__":
    main()