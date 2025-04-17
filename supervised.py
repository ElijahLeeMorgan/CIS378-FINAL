import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Linear, ReLU, Softmax, Sequential
import torch.optim as optim

# For demo, sadly I don't have time to re-write all of this.
import interaction
import gameRead
import time

'''
Notes:

As per Bobeldyk's suggestion, we can use supervised learning to train our model, and use reinforment learning to fine-tune it.
This will give our reinforcement learning model a good starting point, and save training time.

The model will be a simple feedforward neural network, and subject to change. This doesn't matter as long as there's 26 input and 4 output neurons.
In addion, there should be a softmax layer at the end to normalize the output.
'''
# AI Assisted Code
class SimpleNN(Module):
    def __init__(self, input_size:int=6, output_size:int=4):
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
    
    def save_model(self, path:str="supervisedModel.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

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
    def __init__(self, dataPath:str="./dataEngineering/100Attempts-corruptedSickles.csv", model:SimpleNN=None) -> None:
        self.dataPath = dataPath
        self._game_states = None # tuple[tuple[torch.Tensor]
        self._actions = None     # np.ndarray[float]
        self.model = model

        # Set class _dataLoader object
        self._dataLoader = DataLoader()

        # Initialize model
        if self.model is None:
            #self.model = SimpleNN(input_size=len(self._game_states), output_size=len(self._actions)) 
            # Typically 26, but I messed up the sickleXY data while recording. May re-record if nesscessary.
            self.model = SimpleNN(input_size=6, output_size=4)  # Example model initialization, hardcoded input size due to weird data.

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
                self.model.save_model(f"./models/supervisedModel-{currentIndex}.pth")  # Save the model every 1000 epochs

                # Debugging output
                print("Target shape: ", target.shape)
                print("Target data: ", target)
                print("Predictions shape: ", predictions.shape)
                print("Predictions data: ", predictions)

            self.optimizer.zero_grad()
            
            predictions = self.model(tensor)  # Forward pass
            predictions = predictions.softmax(dim=-1)  # Apply softmax to get action probability. Also normalizes the output (because of the weird way I've done this).
            
            targets = {
               0.0: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Move Up
               1.0: torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),  # Move Left
               2.0: torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),  # Move Right
               3.0: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),  # Do Nothing
            }

            if self._actions[currentIndex] not in targets:
                #NOTE Somehow the escape key got into the data. Wrote this as a precaution against further errors.
                print(f"Action {self._actions[currentIndex]} not found in targets. Skipping...")
                currentIndex += 1
                continue
            target = targets[self._actions[currentIndex]]  # Get the target action tensor
            
            loss = self.loss_function(predictions, target)  # Compute loss, should decrease as the model learns
            loss.backward()  # Backpropagation
            
            self.optimizer.step()  # Update weights
            print(f"Epoch {currentIndex + 1}, Loss: {loss.item()}")
            currentIndex += 1
        
        print("Training complete.")
        # Save the model
        self.model.save_model(f"./models/supervisedModel-{currentIndex}.pth") 
        print("Model saved.")

class ModelTester: #NOTE WIP. If I have more time I will implemnt real testing.
    def __init__(self, model:SimpleNN=None) -> None:
        self.model = model

    def test(self, input:torch.Tensor) -> torch.Tensor:
        # Test the model with the given input
        if self.model is None:
            raise ValueError("Model must be loaded before testing.")
        return self.model.predict(input)

class Demo:
    def __init__(self, interaction:interaction.GameInteraction, data:gameRead.GameInfo, modelPath:str=None) -> None:
        self.interaction = interaction
        self.data = data
        self.model = SimpleNN(input_size=6, output_size=4)  # Initialize the model with the correct input and output sizes
        if modelPath:
            self.model.load_state_dict(torch.load(modelPath), strict=False)  # Load the state dictionary into the model
            self.model.eval()  # Set the model to evaluation mode

        # After doing some demos, it looked like we were suffering the same issues as the reinforcement learning model.
        # Thankfully, this isn't the case. The model is actually working as intended, but needs way more training.
            # Check if the internal parameters actually changed
        #    for name, param in self.model.named_parameters():
        #        print(f"Parameter {name}: {param.data}")

    def start(self, attempts:int=10) -> None:
        #NOTE: Yeah, yeah, yeah, repeated code anti-pattern, but the script was slop to begin with.
        # I'm a lot more impressed with my supervised.py code, but hindsight is 20/20.

        # Run the model for a given number of epochs, but doesn't train it.
        # This is useful for testing the model's performance without updating its weights.
        actions = {
                    0: self.interaction.jump,  # Move Up
                    1: self.interaction.moveLeft,  # Move Left
                    2: self.interaction.moveRight,  # Move Right
                    3: self.interaction.nothing,  # Do Nothing
                }
        timeAlive = 0
        longestTimeAlive = 0

        for a in range(attempts): # Each epoch is one life in the game.
            print(f"Attempt #: {a + 1}/{attempts}")

            time.sleep(1) # Give the game time to restart.
            self.interaction.jump()  # Starts the game
            isAlive = True # Player is alive?

            while isAlive: # Player is Alive? Continue until False.
                self.gameState = self.data.getState()[0:6] #FIXME Hardcoded to 6 values for supervised model.
                self.gameState[0], self.gameState[-2] = self.gameState[-2], self.gameState[0]  # Swap the first and second to last values
                #See recordCSV for more details on broken data collection.
                currentTime = self.gameState[5]  # Current time
                isAlive = self.gameState[0]

                #print(f"Game state: {self.gameState}") # Print the game state for debugging
        
                tensor = torch.tensor(self.gameState, dtype=torch.float32)  # Convert to tensor
                # Get the current state and predict the action probabilities
                # Predict and perform the action
                '''
                prediction = self.model(tensor)  # Forward pass
                prediction = prediction.softmax(dim=-1)  # Apply softmax to get action probability. Also normalizes the output (because of the weird way I've done this).
                '''
                prediction = self.model(tensor).argmax().item()

                actions[prediction]() # Perform the action
                #print("Input Tensor: ", tensor) # Print the input tensor for debugging
                #print(f"UP, LEFT, RIGHT, NONE\nPrediction: {prediction}") # Print the action for debugging
                time.sleep(0.05) # Saves CPU

            print("Time alive:", currentTime, "Longest time alive:", longestTimeAlive)
            timeAlive = 30 - currentTime # Time of death.

            if timeAlive > longestTimeAlive:
                longestTimeAlive = timeAlive
                print(f"Longest time alive: {longestTimeAlive} seconds")
                if timeAlive >= 30: 
                    print("Survived for 30 seconds, YAY!!!!!")
                    break
            else:
                print(f"Time alive: {timeAlive} seconds")

def main(path:str="./dataEngineering/100Attempts-corruptedSickles.csv"):
    # Example usage, set args for real use.
    smallModel = SimpleNN(input_size=6, output_size=4)  # Example model initialization, hardcoded input size due to weird data.
    trainer = Trainer(dataPath=path, model=smallModel)  # Example data path
    trainer.loadTrainingData(removeEmptyCollumn=True)
    trainer.train()

    '''
    trainer.model.save_model("supervisedModelSmall.pth")

    test_input = torch.randn(1, 6)  # Example input

    
    for model in [1000, 2000, 3000, 4000, 10000]:
        modelName = f"supervisedModel-{model}.pth"
        modelPath = f"./models/{modelName}"
        trainer.load_model(modelPath)
        output = trainer.model(test_input)
        print(f"Predicted action probabilities with {modelName}:", output)

    # Load the model for inference
    trainer.load_model("supervisedModelSmall.pth")
    output = trainer.model(test_input)
    print("Final model predicted action probabilities:", output)
    '''

if __name__ == "__main__":
    #main()
    main("./dataEngineering/output_augmented_100_attempts.csv")