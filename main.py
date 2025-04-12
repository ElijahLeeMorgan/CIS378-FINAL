import threading
from os import path
from subprocess import run
from time import sleep

import gameRead
import interaction
import train

if __name__ == "__main__":
    #global g
    gameReader = gameRead.GameInfo()
    gameInteractor = interaction.GameInteraction()
    modelTrainer = train.Trainer(gameInteractor, gameReader, model=None)
    # Set model to a LOADED model if you have one to train/demo.
    #TODO Make a model loader system flag to load a model from a file.

    gamepath = './game/SickleDodge.love'

    data_thread = threading.Thread(target=gameRead.listenForData, args=(gameReader,))
    data_thread.daemon = True  # Ensure the thread exits when the main program exits
    data_thread.start()


    # Check if the game file exists
    if path.isfile(gamepath):
        print("Game file found. Proceeding...")
        # Start the game process with the full path
        run(["love", gamepath])
    else:
        print("Game file not found. Please run the build script in the ./game directory.")
        # I tried to use subprocess to run the build script, but the LOVE2D engine wasn't a fan.
        exit(1)

    sleep(3)  # Wait for the game to load
    modelTrainer.train(epochs=100)  # Train the model for 100 epochs
        

    # Check if CUDA is available and set device accordingly
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

