from time import sleep
from interaction import GameInteraction
from gameRead import GameInfo
from warnings import warn

import torch
from torch.nn import Sequential, Linear, ReLU, Softmax
import torch.optim as optim

class trainer(object):
    def __init__(self, interaction:GameInteraction=None, data:GameInfo=None, model=None):
        # Error checking, allows for reusing objects when possible.
        if interaction is None:
            warn("Interaction object cannot be None. Generating new object.")
            self.interaction = GameInteraction("Sickle Dodge")
        else:
            if not isinstance(interaction, GameInteraction):
                raise TypeError("interaction must be an instance of GameInteraction")
            self.interaction = interaction

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
        self.stateSize = len(self.data.getState()) + 26 # 13 sickles max, 2 values each (x,y). Total of 32 values. #FIXME Find actual max sickles.
        self.gameState = self.data.getState()

        if model is None:
            print("Model is None, generating new model.")
            self.model = Sequential(
                Linear(self.data.state_size, 32),
                ReLU(),
                Linear(32, 16),
                ReLU(),
                Linear(16, 8),
                ReLU(),
                Linear(8, self.actionSize),
                Softmax(dim=-1)
            )
        else:
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must be an instance of torch.nn.Module")
            self.model = model
    
    def penalize(self, penalty:float):
        # Placeholder for penalty logic
        # This could be a negative reward or a loss function adjustment
        pass

    def reward(self, reward:float):
        # Placeholder for reward logic
        # This could be a positive reward or a loss function adjustment
        pass

    def train(self):
        # Check if the game is over, and restart if so.
        timeAlive = 0
        lastTimeAlive = 0
        longestTimeAlive = 0
        
        while timeAlive < 30:
            timeAlive = self.data.getState[5]  # Timer
            path = "./models/"

            if self.data.getState[0] == False: # isAlive?

                if timeAlive > lastTimeAlive:
                    lastTimeAlive = timeAlive
                    #TODO Reward the model for last time alive. Seperate function.
                    #self.reward(timeAlive - lastTimeAlive)
                    print(f"Time alive: {timeAlive} seconds")
                else:
                    ...
                    #TODO Penalize the model for dying too soon. Seperate function.
                    #self.penalize(timeAlive - lastTimeAlive)
                    print(f"Time alive: {timeAlive} seconds")

                if timeAlive > longestTimeAlive:
                    # Optional, but useful for debugging and seeing trainign progress.
                    longestTimeAlive = timeAlive
                    print(f"Longest time alive: {longestTimeAlive} seconds")

                    #TODO Save the model if it is the longest time alive. Handle model saving in a different function.
                    #self.save_model(path + "model{longestTimeAlive}.pth")
                    #self.model.save_model("model{longestTimeAlive}.pth")
                    print(f"Model saved to {path}.")
                    
                
                
                self.interaction.jump()  # Restarts the game
                continue

            # Wait for the next frame
            sleep(0.05)

    def save_model(self, path: str):
        # Placeholder for model saving logic
        pass

    def load_model(self, path: str):
        # Placeholder for model loading logic
        pass

if __name__ == "__main__":
    ...
    #training_loop()