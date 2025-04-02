import torch
import socket
import os

def listenForData():
    # Create a UDP socket
    ip = "127.0.0.1"
    port = 12345
    buffer_size = 1024

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    print(f"Listening for data on {ip}:{port}")

    while True:
        # Receive data from the socket
        data, addr = sock.recvfrom(buffer_size)
        print(f"Received data from {addr}: {data.decode('utf-8')}")

        # Here you can process the received data as needed

def checkForGame() -> bool:
    return os.path.isfile('./game/SickleDodge.love')

if __name__ == "__main__":
    # Check if the game file exists
    if not checkForGame():
        print("Game file not found. Please ensure you built the game first.")
        exit(1)

    # Check if CUDA is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

