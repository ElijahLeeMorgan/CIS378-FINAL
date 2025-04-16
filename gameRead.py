import socket
from luadata import unserialize
from os import path

class GameInfo:
    def __init__(self):
        self.isAlive: bool = True

        self.playerX: float = 0.0
        self.playerY: float = 0.0
        self.playerVelocityX: float = 0.0
        self.playerVelocityY: float = 0.0

        self.timer: int = 30

        self.sickles: list[dict[float, float, bool]] = []           # list[dict[x:float,y:float, is_alive:bool]]

        self.dataSize: int = 10 + len(self.sickles) * 2  # 4 for the player position and velocity, 2 for max number of sickles (x,y)

    def __str__(self):
        return (f"Player Alive?:\t\t{self.isAlive,} {type(self.isAlive)}\n"
                f"Player Position:\t({self.playerX}, {self.playerY}), {type(self.playerX)}\n"
                f"Player Velocity:\t({self.playerVelocityX}, {self.playerVelocityY}, {type(self.playerVelocityX)})\n"
                f"Timer:\t\t\t{self.timer}, {type(self.timer)}\n"
                f"Number of Sickles:\t{len(self.sickles)}, {type(self.sickles)}\n")

    # Start AI Assisted Code
    def estimateVelocity(self, oldX, oldY, newX, newY) -> tuple[float, float]:
        deltaX = newX - oldX
        deltaY = newY - oldY

        # Calculate the velocity
        velocityX = deltaX / 0.016  # Assuming a frame rate of 60 FPS
        velocityY = deltaY / 0.016
        return velocityX, velocityY
    
    def update(self, is_alive, playerX, playerY, timer, sickles) -> None:
        self.playerVelocityX, self.playerVelocityY = self.estimateVelocity(self.playerX, self.playerY, playerX, playerY)
        self.playerX = playerX
        self.playerY = playerY
        # Update the sickles list with new data, may be difficult especially when sickles are added or removed.
        self.sickles = sickles

        self.dataSize = 10 + len(self.sickles) * 2  # 4 for the player position and velocity, 2 for max number of sickles (x,y)

        self.timer = timer #NOTE timer bug is not from the data retrieval.
        self.isAlive = is_alive

    def getState(self) -> list[float]:
        state = [float(self.isAlive), self.playerX, self.playerY, self.playerVelocityX, self.playerVelocityY, float(self.timer)]
        for sickle in self.sickles:
            #NOTE These should alreayd be floats, but just in case.
            #if sickle['is_alive']: # Skips dead sickles. #FIXME isAlive not appended to data ahead of time. Not worh fixing right now.
            state.append(sickle['x'])
            state.append(sickle['y'])
        return state

def listenForData(GINFO: GameInfo, ip:str="127.0.0.1", port:int=12345, buffer_size:int=1024, saveToFile=False) -> None:
    # Create a UDP socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    #print(f"Listening for data on {ip}:{port}")

    while True:
        # Receive data from the socket
        data, addr = sock.recvfrom(buffer_size)
        raw_data = data.decode('utf-8')
        # Parse the received data into dict with keys ["player"], ["sickles"], ["timer"]
        gameData = unserialize(raw_data, encoding="utf-8", multival=False)

        GINFO.update(
            is_alive=gameData['player']['is_alive'],
            playerX=gameData['player']['x'],
            playerY=gameData['player']['y'],
            timer=gameData['timer'],
            sickles=gameData['sickles']
        )

        #print(f"Received data from {addr}:\n{GINFO}")

        # Optionally save the data to a file
        if saveToFile:
            # Check if the file exists, if not create it
            if not path.exists("./game_data.txt"):
                with open("./game_data.txt", "w") as f:
                    f.write(f"{GINFO}\n")
            else:
                # Append the data to the file
                with open("./game_data.txt", "a") as f:
                    f.write(f"{GINFO}\n")
# End AI Assisted Code
