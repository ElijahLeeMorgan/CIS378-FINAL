from time import sleep
from interaction import GameInteraction
from gameRead import GameInfo
from warnings import warn
import numpy as np
from os import path, makedirs

import torch
from torch.nn import Sequential, Linear, ReLU, Softmax
import torch.optim as optim

# AI Assited Code
class Trainer():
    def __init__(self, interaction:GameInteraction=None, data:GameInfo=None, model:torch.nn.Module=None):
        # Error checking, allows for reusing objects when possible.

        # Window interaction Class
        if interaction is None:
            warn("Interaction object cannot be None. Generating new object.")
            self.interaction = GameInteraction("Sickle Dodge")
        else:
            if not isinstance(interaction, GameInteraction):
                raise TypeError("interaction must be an instance of GameInteraction")
            self.interaction = interaction

        # Game data reading Class
        if data is None:
            warn("GameInfo object cannot be None. Generating new object.")
            self.data = GameInfo()
        else:
            if not isinstance(data, GameInfo):
                raise TypeError("game must be an instance of GameInfo")
            self.data = data
        
        # Define the action space
        self.actions = {
            0: self.interaction.jump,
            1: self.interaction.moveLeft,
            2: self.interaction.moveRight,
            3: self.interaction.nothing,
        }
        self.actionSize = len(self.actions)
        self.stateSize = len(self.data.getState()) + 18 # 9 sickles max, 2 values each (x,y). Total of 26 values.
        #TODO Maybe add last action (button pressed) to the state? This would allow for better prediction of the next action.
        #NOTE If I do this I need to adjust functions that read the self.gameState variable. 
        self.gameState = self.data.getState()

        if model is None:
            print("Model is None, generating new model.")
            self.model = Sequential(
            Linear(26, 16), #FIXME Hardcoded input size. Should be self.stateSize <- Buggy, doesn't consider when sickles are missing. 
            ReLU(),
            Linear(16, 8),
            ReLU(),
            Linear(8, self.actionSize), # Currently 4 actions.
            Softmax(dim=-1)
            )
        else:
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must be an instance of torch.nn.Module")
            self.model = model
        
        # Initialize the optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1) # Learning rate decay, optional but allows for better training.


    def _currentLearningRate(self) -> float:
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        return current_lr

    def _penalize(self) -> None:
        # Apply a penalty by adjusting the model's loss function

        # Generate a dummy target tensor with zeros (penalty scenario)
        target = torch.zeros(self.actionSize)
        target = target.unsqueeze(0)  # Add batch dimension
        
        # Get the current state and predict the action probabilities
        state = self._tensorData(self.gameState) #FIXME turn into a object var to save CPU cost.
        predictions = self.model(state)
        
        # Calculate the loss (penalty)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predictions, target)
        
        # Backpropagate the penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Penalty applied.")

    def _reward(self, reward: float) -> None:
        # Apply a reward by adjusting the model's loss function
        
        # Generate a dummy target tensor with a positive reward
        target = torch.full((1, self.actionSize), reward / self.actionSize)
        
        # Get the current state and predict the action probabilities
        predictions = self.model(self._tensorData(self.gameState)) #FIXME turn into a object var to save CPU cost.
        
        # Calculate the loss (reward)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predictions, target)
        
        # Backpropagate the reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Reward: {reward}")

    def _tensorData(self, data: list) -> torch.Tensor:
        # Data is alreayd a list of floats, so we can just convert it to a tensor.
        # I didn't realize this when I wrote the code below.
        '''
        for i in data:
            if isinstance(i, (list, tuple)):
                arr = np.append(arr, np.ravel(i))  # Flatten and append
                #NOTE No need to expect non-numeric data here, as the game should only send lists of numeric data.
            else:
                try:
                    arr = np.append(arr, float(i))  # Append single value
                    #NOTE This will work on all but strs, but that shouldn't be a problem.
                except ValueError:
                    raise TypeError(f"Data type {type(i)} not supported.")
        '''
        print("StateSize:", self.stateSize)
        if len(data) < 26:
            data = np.vectorize(float)(data)  # Convert to float
            data = np.pad(data, (0, 26 - len(data)), 'constant', constant_values=0)
        data = data.astype(np.float32)  # Ensure all values are signed floats
        return torch.tensor(data).unsqueeze(0)  # Convert to tensor and add batch dimension

    def train(self, epochs: int = 1) -> None:
        timeAlive = 0
        lastTimeAlive = 0
        longestTimeAlive = 0

        for epoch in range(epochs): # Each epoch is one life in the game.
            print(f"Epoch {epoch + 1}/{epochs}")

            sleep(3) # Give the game time to restart.
            
            #for batch in train_loader: # Replace with real-time data fetching
            # Perform forward pass, compute loss, and backpropagation
            self.optimizer.zero_grad()
            #loss = compute_loss(batch) # Placeholder for loss computation, replace with actual logic..?
            #loss.backward()
            self.optimizer.step()

            # Data order:
            #self.isAlive, self.playerX, self.playerY, self.playerVelocityX, self.playerVelocityY, self.timer, SickleX, SkicleY, SickleX2, SicklyY2, ...
            
            self.interaction.jump()  # Starts the game
            isAlive = True # Player is alive?

            while isAlive: # Player is Alive? Continue until False.
                self.gameState = self.data.getState() # list[float]
                currentTime = self.gameState[1]  # Current time
                isAlive = self.gameState[0]

                print(f"Game state: {self.gameState}") # Print the game state for debugging
        
                tensorState = self._tensorData(self.gameState)  # Convert to tensor and add batch dimension
                print(f"State: {tensorState}") #FIXME: This is the last thing that will print. WHY?
                # Get the current state and predict the action probabilities
                # Predict and perform the action
                
                #FIXME : RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x6 and 22x26)
                print("Tensor state shape:", tensorState.shape)
                
                action = self.model(tensorState).argmax().item()  # Get the action with the highest probability according to the model.
                
                self.actions[action]()  #FIXME No idea if this will work lol. It should run the returned function from the model.
                print(f"Action: {action}")

            timeAlive = currentTime # Time of death.
            if timeAlive > longestTimeAlive:
                longestTimeAlive = timeAlive
                print(f"Longest time alive: {longestTimeAlive} seconds")
                # Save model.
                self._saveModel()
                #if timeAlive >= 30: #FIXME Gamestate is updating with values like 145 after 2 seconds.
                #    print("Survived for 30 seconds, stopping training.")
                #    break # If the player survives for 30 seconds, stop training.
                #NOTE do not 'contnue' here, we still need to reward the model for time alive.

            if timeAlive > lastTimeAlive:
                self._reward((timeAlive - lastTimeAlive) / 30) # Divide by 30 to normalize the reward float.
                # Normally, I would max to 30 to prevent accidental penalty, but we know this will only run if timeAlive > lastTimeAlive.
                print(f"Time alive: {timeAlive} seconds")
            else:
                self._penalize() # Penalize for dying too soon.
                print(f"Time alive: {timeAlive} seconds")
            lastTimeAlive = timeAlive # Update previous time alive.
            
            # Step the scheduler at the end of each epoch
            self.scheduler.step()
            # Optional: Print the current learning rate
            self._currentLearningRate()

    def _saveModel(self, filePath:str="./models/"):
        if not path.exists(filePath):
            print(f"Creating directory {filePath}...")
            makedirs(filePath)
        torch.save(self.model.state_dict(), filePath + "model.pth")
        print(f"Model saved to {filePath}.")

            
    def load_model(self, path: str):
        # Placeholder for model loading logic
        pass

if __name__ == "__main__":
    ...
    #training_loop()