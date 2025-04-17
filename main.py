import threading
from os import path
from subprocess import Popen
from time import sleep

import gameRead
import interaction
import train
import supervised

if __name__ == "__main__":
    gameReader = gameRead.GameInfo()
    # Set model to a LOADED model if you have one to train/demo.
    supervisedDemo = True

    gamepath = './game/SickleDodge.love'

    data_thread = threading.Thread(target=gameRead.listenForData, args=(gameReader,))
    data_thread.daemon = True  # Ensure the thread exits when the main program exits
    data_thread.start()


    # Check if the game file exists
    if path.isfile(gamepath):
        print("Game file found. Proceeding...")
        # Start the game process with the full path
        game = Popen(["love", gamepath])
        print("Game process started. Waiting for it to load...")
        sleep(3)  # Give the game a moment to start
        
        # Wait for the game process to be ready. #FIXME Needs works.
        '''
        while game.poll() is None:
            print("Waiting for the game to load...")
            sleep(1)
        '''
    else:
        print("Game file not found. Please run the build script in the ./game directory.")
        # I tried to use subprocess to run the build script, but the LOVE2D engine wasn't a fan.
        exit(1)
    
    #NOTE It's very likely that the char will die before the game is ready. This should be fine.
    gameInteractor = interaction.GameInteraction("Sickle Dodge")
    print("Game interaction object created.")

    print("Trainer object created. Training model...")
    if supervisedDemo:
        modelDemo = supervised.Demo(gameInteractor, gameReader, modelPath="./models/supervised/supervisedModel-134180.pth")
        modelDemo.start(attempts=100)
    else:
        modelTrainer = train.Trainer(gameInteractor, gameReader, model=None)
        modelTrainer.train(epochs=100)  # Train the model for 100 epochs

    # Check if CUDA is available and set device accordingly
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
