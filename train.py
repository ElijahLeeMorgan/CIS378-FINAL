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
        self.stateSize = len(self.data.getState()) + 16 # 8 sickles max, 2 values each (x,y). Total of 24 values.
        self.gameState = self.data.getState()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1) # Learning rate decay, optional but allows for better training.

        if model is None:
            print("Model is None, generating new model.")
            self.model = Sequential(
            Linear(self.data.state_size, 24),
            ReLU(),
            Linear(24, 16),
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
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def currentLearningRate(self) -> float:
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        return current_lr

    def penalize(self) -> None:
        # Apply a penalty by adjusting the model's loss function

        # Generate a dummy target tensor with zeros (penalty scenario)
        target = torch.zeros(self.actionSize)
        target = target.unsqueeze(0)  # Add batch dimension
        
        # Get the current state and predict the action probabilities
        state = torch.tensor(self.data.getState(), dtype=torch.float32).unsqueeze(0)
        predictions = self.model(state)
        
        # Calculate the loss (penalty)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predictions, target)
        
        # Backpropagate the penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Penalty applied.")

    def reward(self, reward: float) -> None:
        # Apply a reward by adjusting the model's loss function
        
        # Generate a dummy target tensor with a positive reward
        target = torch.full((1, self.actionSize), reward / self.actionSize)
        
        # Get the current state and predict the action probabilities
        state = torch.tensor(self.data.getState(), dtype=torch.float32).unsqueeze(0)
        predictions = self.model(state)
        
        # Calculate the loss (reward)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predictions, target)
        
        # Backpropagate the reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Reward: {reward}")

    def train(self, epochs: int = 100) -> None:
        # Check if the game is over, and restart if so.
        timeAlive = 0
        lastTimeAlive = 0
        longestTimeAlive = 0
        path = "./models/"

        for epoch in range(epochs):
            #for batch in train_loader: # Replace with real-time data fetching
            # Perform forward pass, compute loss, and backpropagation
            self.optimizer.zero_grad()
            #loss = compute_loss(batch) # Placeholder for loss computation, replace with actual logic..?
            #loss.backward()
            self.optimizer.step()

            #self.isAlive, self.playerX, self.playerY, self.playerVelocityX, self.playerVelocityY, self.timer, SickleX, SkicleY, SickleX2, SicklyY2, ...
            
            self.gameState = self.data.getState()
            isAlive = self.gameState[0]
            currentTime = self.gameState[1]  # Current time

            if isAlive == False: # isAlive?
                timeAlive = currentTime

                if timeAlive > longestTimeAlive:
                    longestTimeAlive = timeAlive
                    print(f"Longest time alive: {longestTimeAlive} seconds")
                    # Save model.
                    torch.save(self.model.state_dict(), path + "model.pth")
                    print(f"Model saved to {path}.")
                    #NOTE do not 'contnue' here, we still need to reward the model for time alive.

                if timeAlive > lastTimeAlive:
                    self.reward((timeAlive - lastTimeAlive) / 30) # Divide by 30 to normalize the reward float.
                    # Normally, I would max to 30 to prevent accidental penalty, but we know this will only run if timeAlive > lastTimeAlive.
                    print(f"Time alive: {timeAlive} seconds")
                else:
                    self.penalize() # Penalize for dying too soon.
                    print(f"Time alive: {timeAlive} seconds")
                lastTimeAlive = timeAlive # Update previous time alive.

                
            
            self.interaction.jump()  # Restarts the game

            # Wait for the next frame
            sleep(0.05)
            

            
            # Step the scheduler at the end of each epoch
            self.scheduler.step()
            # Optional: Print the current learning rate
            self.currentLearningRate()
        
        

        

    def save_model(self, path: str):
        # Placeholder for model saving logic
        pass

    def load_model(self, path: str):
        # Placeholder for model loading logic
        pass

if __name__ == "__main__":
    ...
    #training_loop()