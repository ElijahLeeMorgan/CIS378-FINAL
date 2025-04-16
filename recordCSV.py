import threading
import os
from datetime import datetime
from time import sleep
import gameRead
import keyboard

'''
Standalone script to record game data to a CSV file.

- Must be run in the same directory as gameRead.py.
- The game must be running in the background or you will receive an error.
  - You can run the game via "love game/SickleDodge.love" in the terminal.

Output CSV files are saved in the current directory, and are not postpended with 0's in empty cells.
This was an oversight, but will be circumvented in the supervised learning preprocessing.
'''

# AI Assisted Code
def _createCSV(fileName:str="gameRecording") -> None|str:
    # Create a CSV file with the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fileName = f"{fileName}_{current_time}.csv"
    
    # Check if the file already exists
    if os.path.exists(fileName):
        print(f"File {fileName} already exists. Please choose a different name.")
        return None
    
    # Create the CSV file, write headers
    with open(fileName, 'w') as f:
        headers = ["isAlive", "playerX", "playerY", "playerVelocityX", "playerVelocityY", "timer"]
        sickle_headers = [f"sicklesX{i},sicklesY{i}" for i in range(14)] #FIXME sickles max: 14..? That's weird, is my game recording class right?
        all_headers = headers + sickle_headers + ["keyboardAction"]
        f.write(",".join(all_headers) + "\n")
    
    return fileName

def _appendCSV(fileName:str, data:list) -> None:
    # Check if the file exists
    if not os.path.exists(fileName):
        print(f"File {fileName} does not exist. Generating a new file.")
        fileName = _createCSV(fileName)
    
    # Append data to the CSV file
    with open(fileName, 'a') as f:
        f.write(",".join(map(str, data)) + "\n")

def _currentKeyboardInput() -> float:
    # Check if specific keys are pressed and return their corresponding actions
    '''
    Actions are eventually conveted to a float value for use in a tensor.
    self.actions = {
            0.0: up/spacebar
            1.0: leftArrow
            2.0: rightArrow
            3.0: everything else
        }
    '''
    if keyboard.is_pressed('space'):
        return 0.0
    elif keyboard.is_pressed('up'):
        return 0.0
    elif keyboard.is_pressed('left'):
        return 1.0
    elif keyboard.is_pressed('right'):
        return 2.0
    elif keyboard.is_pressed('escape'):
        print("Exiting...")
        return -1.0
    else:
        return 3.0 # Aka no action/nothing.

def recordGameData(fileName:str="gameRecording") -> str:
    # Create a reader object.
    gameReader = gameRead.GameInfo()
    
    # Create a CSV file
    fileName = _createCSV(fileName)

    # Record game data
    data_thread = threading.Thread(target=gameRead.listenForData, args=(gameReader,))
    data_thread.daemon = True  # Ensure the thread exits when the main program exits
    data_thread.start()
    _keyboardInput = 3.0

    while _keyboardInput != -1.0: # ESC key pressed.
        # Get the current game state
        game_state = gameReader.getState()

        # Postpend 0's up to collumn 28(14 sickleXY) + 6(gamedata) + 1 (keyboardOutput) = 35 if the game_state is too short.
        # Remember, the last value is removed before converting to a tensor.
        # Keeps keyboard input at the end of the list, and ensure tensor size is consistent.
        # Error checking will still occour in the supervised learning preprocessing just in case.
        while (len(game_state) - 6) < 28: #FIXME Hardcoded, should make csv always have 35 collumns.
            game_state.append(0.0)
        _keyboardInput = _currentKeyboardInput()  # Get the current keyboard input
        game_state.append(_keyboardInput)  # Append the current keyboard input as the float action conversion.
        
        if game_state[0] == 1.0: # Player is alive
            # Append the game state to the CSV file
            _appendCSV(fileName, game_state)
        
        # Sleep for a short duration to avoid excessive CPU usage
        sleep(0.08) # Approximately twice a frame at 60 FPS.
    
    print(f"Recording stopped. Data saved to {fileName}.")
    return fileName

def main(gamepath:str='./game/SickleDodge.love') -> None:

    # Check if the game is running
    if not os.path.exists(gamepath):
        print("Game file not found. Please run the build script in the ./game directory.")
        exit(1)
    
    # Start recording game data
    recordGameData()

if __name__ == "__main__":
    main()
