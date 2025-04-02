import torch
import socket
import threading
import os
import json

class GameInfo:
    def __init__(self):
        self.isAlive = True

        self.playerX = 0
        self.playerY = 0
        self.playerVelocityX = 0
        self.playerVelocityY = 0

        self.timer = 30

        self.sickles = []

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
        self.sickles = sickles

def listenForData(GINFO: GameInfo, ip:str="127.0.0.1", port:int=12345, buffer_size:int=1024):
    # Create a UDP socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    print(f"Listening for data on {ip}:{port}")

    while True:
        # Receive data from the socket
        data, addr = sock.recvfrom(buffer_size)
        raw_data = data.decode('utf-8')

        try:
            # Parse the received data as JSON
            parsed_data = json.loads(raw_data)

            # Extract player information
            player_info = parsed_data.get("player", {})
            player_x = player_info.get("x", 0)
            player_y = player_info.get("y", 0)
            player_is_alive = player_info.get("is_alive", True)

            # Extract sickles information
            sickles_info = parsed_data.get("sickles", {})
            sickles = [
                {"x": sickle.get("x", 0), "y": sickle.get("y", 0)}
                for sickle in sickles_info.values()
            ]

            # Extract timer information
            timer = parsed_data.get("timer", 0)

            GINFO.update(player_is_alive, player_x, player_y, timer, sickles)

            # Print parsed data for debugging
            print(f"Player: x={player_x}, y={player_y}, is_alive={player_is_alive}")
            print(f"Sickles: {sickles}")
            print(f"Timer: {timer}")



        except json.JSONDecodeError:
            print(f"Failed to parse data from {addr}: {raw_data}")


def checkForGame() -> bool:
    return os.path.isfile('./game/SickleDodge.love')

if __name__ == "__main__":
    g = GameInfo()

    data_thread = threading.Thread(target=listenForData, args=(g,))
    data_thread.daemon = True  # Ensure the thread exits when the main program exits
    data_thread.start()


    # Check if the game file exists
    if not checkForGame():
        print("Game file not found. Please ensure you built the game first.")
        exit(1)

    # Check if CUDA is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

