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
            Linear(64, 128),  # Hidden layer
            ReLU(),
            Linear(128, 128),  # Hidden layer
            ReLU(),
            Linear(128, 64),  # Hidden layer
            ReLU(),
            Linear(64, output_size),  # Output layer with actionSize neurons (4 actions)
            Softmax(dim=-1)  # Softmax to normalize action probabilities.
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Forward pass through the network. Take the max output for predicted action (by index).
        return self.model(x)
    
    def backward(self, loss:torch.Tensor) -> None:
        # Backward pass to compute gradients
        loss.backward()

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        # Predict the action probabilities
        with torch.no_grad():
            probs = self.model(x)
            print("Predicted action probabilities:", probs)
            return probs

# Example Data
#game_states = [[1.0, 50.0, 100.0, 0.0, 0.0, 30.0, ...], [1.0, 60.0, 90.0, 0.0, 0.0, 29.0, ...]]
#actions = [0.0, 2.0]  # Corresponding actions
class DataLoader:
    def __init__(self):
        pass

    def _csvToNumpy(self, csv_file:str, removeEmptyCollumn:bool=False) -> tuple[np.ndarray[float], np.ndarray[float]]: # Mat (2d ndarray), Vec (1d ndarray)
        # Load CSV data
        # gameData-26floats, action-1float. In order by row.
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        # Split into game states and actions
        if removeEmptyCollumn:
            data = data[:, np.any(data != 0, axis=0)]  # Remove columns with all zeros
            print("Empty columns removed, new shape RxC: ", data.shape)
        game_states = np.delete(data, -1, axis=1)  # game states (all columns except the last)
        actions = np.take(data, -1, axis=1)  # output actions (last column)
        return game_states, actions

    def _preprocess_data(self, game_states:np.ndarray[float], actions:np.ndarray[float]) -> tuple[np.ndarray[torch.Tensor], np.ndarray[float]]:
        
        game_states = torch.tensor(game_states, dtype=torch.float32)  # Convert to one large tensor
        game_states = torch.unbind(game_states, dim=0)  # Unbind to get a list of row-tensors

        # Should already be a 1D ndarray
        #actions = np.ndarray(actions, dtype=torch.float32)  # Actions as floats

        print("Game states length: ", len(game_states))
        print("Game states shape: ", game_states[0].shape)
        print("Actions shape: ", actions.shape)
        return game_states, actions
    
    def getdata(self, csv_file:str, removeEmptyCollumn:bool=False) -> tuple[tuple[torch.Tensor], np.ndarray[float]]:
        # Load data
        game_states, actions = self._csvToNumpy(csv_file, removeEmptyCollumn)
        # Preprocess data
        game_states, actions = self._preprocess_data(game_states, actions)
        return game_states, actions

class Trainer:
    def __init__(self, dataPath:str="../dataEngineering/100Attempts-corruptedSickles.csv", model:SimpleNN=None) -> None:
        self.dataPath = dataPath
        self._game_states = None # tuple[tuple[torch.Tensor]
        self._actions = None     # np.ndarray[float]

        # Set class _dataLoader object
        self._dataLoader = DataLoader()

        # Initialize model
        if model is None:
            self.model = SimpleNN(input_size=len(self._game_states), output_size=4) # Typically 26, but I messed up the sickleXY data while recording. May re-record if nesscessary.
        else:
            self.model = model

        # Define loss and optimizer
        self.loss_function = nn.MSELoss()  # Mean Squared Error loss for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def loadTrainingData(self, removeEmptyCollumn:bool=False) -> None:
        self._game_states, self._actions = self._dataLoader.getdata(self.dataPath, removeEmptyCollumn) # tuple[tuple[torch.Tensor], np.ndarray[float]]
        if self._game_states is None or self._actions is None:
            raise ValueError("Game states and actions must be loaded before training.")

    def train(self) -> None:
        '''
        Each tensor in _game_states is a 1D tensor of floats, representing the game state.
        Each action in _actions is a 1D tensor of floats, representing the action taken by a real human player.
        The model will be trained to predict the action taken by the player given the game state.
        The model's output will be a 1D tensor of floats, representing the action taken by the player.

        If the model guesses the action correctly, it will be rewarded with a positive value.
        '''
        if self._game_states is None or self._actions is None:
            raise ValueError("Game states and actions must be loaded before training.\nPlease call loadTrainingData() first.") 

        currentIndex = 0
        print("Game states and actions loaded. Training...")
        # Supervised Training loop

        for tensor in self._game_states:
            tensor.requires_grad = True

            if currentIndex % 1000 == 0 and currentIndex != 0:
                print(f"Training on game state {currentIndex}/{len(self._game_states)}")
                self.save_model(f"../models/supervisedModel-{currentIndex}.pth")  # Save the model every 1000 epochs

                # Debugging output
                print("Target shape: ", target.shape)
                print("Predictions shape: ", predictions.shape)

            self.optimizer.zero_grad()
            
            predictions = self.model(tensor)  # Forward pass
            
            # This is almost certainly wrong, I probably have to max the correct index on a 4x1 (1x4..?) tensor. 
            # NOT make a tensor of the action.
            #self.actions = {
            #   0: self.interaction.jump,
            #   1: self.interaction.moveLeft,
            #   2: self.interaction.moveRight,
            #   3: self.interaction.nothing,
            #}
            #
            target = torch.tensor(self._actions[currentIndex], dtype=torch.float32).unsqueeze(0)  # Convert to tensor and match shape
            
            loss = self.loss_function(predictions, target)  # Compute loss
            loss.backward()  # Backpropagation
            
            self.optimizer.step()  # Update weights
            print(f"Epoch {currentIndex + 1}, Loss: {loss.item()}")
            currentIndex += 1
        
        print("Training complete.")
        # Save the model
        self.save_model(f"../models/supervisedModel-{currentIndex}.pth") 
        print("Model saved.")
        
    def save_model(self, path:str="supervisedModel.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path:str="supervisedModel.pth"):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


class modelTester: #NOTE WIP. If I have more time I will implemnt real testing.
    def __init__(self, model:SimpleNN=None) -> None:
        self.model = model

    def test(self, input:torch.Tensor) -> torch.Tensor:
        # Test the model with the given input
        if self.model is None:
            raise ValueError("Model must be loaded before testing.")
        return self.model.predict(input)


def main():
    # Example usage, set args for real use.
    smallModel = SimpleNN(input_size=6, output_size=4)  # Example model initialization, hardcoded input size due to weird data.
    trainer = Trainer(dataPath="../dataEngineering/100Attempts-corruptedSickles.csv", model=smallModel)  # Example data path
    trainer.loadTrainingData(removeEmptyCollumn=True)
    trainer.train()

    trainer.save_model("supervisedModelSmall.pth")

    test_input = torch.randn(1, 6)  # Example input

    for model in [1000, 2000, 3000, 4000, 10000]:
        modelName = f"supervisedModel-{model}.pth"
        modelPath = f"../models/{modelName}"
        trainer.load_model(modelPath)
        output = trainer.model(test_input)
        print(f"Predicted action probabilities with {modelName}:", output)

    # Load the model for inference
    trainer.load_model("supervisedModelSmall.pth")
    output = trainer.model(test_input)
    print("Final model predicted action probabilities:", output)

if __name__ == "__main__":
    main()