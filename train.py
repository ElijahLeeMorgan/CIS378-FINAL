from time import sleep
from interaction import GameInteraction
from gameRead import GameInfo

# Initialize the game
game = GameInteraction()

def training_loop(game:GameInfo) -> bool: # Returns True once model training is finished.
    char = GameInteraction("Sickle Dodge")

    while True:
        # Check if the game is over
        if game.is_game_over():
            game.jump()  # Restart the game
            continue

        # Wait for the next frame
        sleep(0.05)

if __name__ == "__main__":
    training_loop()