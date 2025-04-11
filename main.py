import threading
from os import path
from subprocess import run

import gameRead
import train

if __name__ == "__main__":
    #global g
    g = gameRead.GameInfo()
    gamepath = './game/SickleDodge.love'

    data_thread = threading.Thread(target=gameRead.listenForData, args=(g,))
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
        

    # Check if CUDA is available and set device accordingly
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

