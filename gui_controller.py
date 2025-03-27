import socket

def send_command(command, host="127.0.0.1", port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(command.encode("utf-8"))
        print(f"Sent command: {command}")

if __name__ == "__main__":
    # Command to set opacity of TF1 to 0.5
    command = "legend delete bottle"  # TF1 corresponds to index 0
    send_command(command)
