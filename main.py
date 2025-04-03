import torch
import socket
import threading
import os
import subprocess
import luadata

class GameInfo:
    def __init__(self):
        self.isAlive = True

        self.playerX = 0
        self.playerY = 0
        self.playerVelocityX = 0
        self.playerVelocityY = 0

        self.timer = 30

        self.sickles = []           # list[dict[x,y, is_alive]]

    def __str__(self):
        return (f"Player Position: ({self.playerX}, {self.playerY})\n"
                f"Player Velocity: ({self.playerVelocityX}, {self.playerVelocityY})\n"
                f"Timer: {self.timer}\n"
                f"Number of Sickles: {len(self.sickles)}")

    # Start AI Assisted Code
    def estimateVelocity(self, oldX, oldY, newX, newY):
        deltaX = newX - oldX
        deltaY = newY - oldY

        # Calculate the velocity
        velocityX = deltaX / 0.016  # Assuming a frame rate of 60 FPS
        velocityY = deltaY / 0.016
        return velocityX, velocityY
    
    def update(self, is_alive, playerX, playerY, timer, sickles):
        self.playerVelocityX, self.playerVelocityY = self.estimateVelocity(self.playerX, self.playerY, playerX, playerY)
        self.playerX = playerX
        self.playerY = playerY
        #TODO: Add player velocity estimation for sickles.
        # Update the sickles list with new data, may be difficult especially when sickles are added or removed.
        self.sickles = sickles

        #TODO: Add sickle velocity estimation.

        self.timer = timer
        self.isAlive = is_alive

def listenForData(GINFO: GameInfo, ip:str="127.0.0.1", port:int=12345, buffer_size:int=1024):
    # Create a UDP socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    print(f"Listening for data on {ip}:{port}")

    while True:
        # Receive data from the socket
        data, addr = sock.recvfrom(buffer_size)
        raw_data = data.decode('utf-8')
        # Parse the received data into dict with keys ["player"], ["sickles"], ["timer"]
        gameData = luadata.unserialize(raw_data, encoding="utf-8", multival=False)

        GINFO.update(
            is_alive=gameData['player']['is_alive'],
            playerX=gameData['player']['x'],
            playerY=gameData['player']['y'],
            timer=gameData['timer'],
            sickles=gameData['sickles']
        )

        print(f"Received data from {addr}:\n{GINFO}")

if __name__ == "__main__":
    #global g
    g = GameInfo()
    gamepath = './game/SickleDodge.love'

    data_thread = threading.Thread(target=listenForData, args=(g,))
    data_thread.daemon = True  # Ensure the thread exits when the main program exits
    data_thread.start()


    # Check if the game file exists
    if os.path.isfile(gamepath):
        print("Game file found. Proceeding...")
        # Start the game process with the full path
        subprocess.run(["love", gamepath])
    else:
        print("Game file not found. Please run the build script in the ./game directory.")
        # I tried to use subprocess to run the build script, but the LOVE2D engine wasn't a fan.
        exit(1)
        

    # Check if CUDA is available and set device accordingly
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

