from ultralytics import YOLO
import socket
import cv2
import numpy as np

# Loading the model
model = YOLO("yolo11n.pt")

# TCP server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(0)
conn, addr = server_socket.accept()

# List to store the received frames
frames = []

while True:
    # Reception of the size of the incoming frame (4 bytes for an integer)
    data = conn.recv(4)
    if not data:
        break
    print("Frame received")
    size = int.from_bytes(data, 'little')

    # Reception of the frame data based on the expected size
    frame_data = b''
    while len(frame_data) < size:
        packet = conn.recv(size - len(frame_data))
        if not packet:
            break
        frame_data += packet

    # Decode the frame from JPEG format
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # Instead of processing immediately, store the frame
    frames.append(frame)

conn.close()
server_socket.close()

# If frames were received, I write them to a video file
if frames:
    # Get frame dimensions from the first frame
    height, width, layers = frames[0].shape
    fps = 30  # Change this to match the source's fps if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = 'received_video.mp4'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for f in frames:
        out.write(f)
    out.release()
    print(f"Video saved as {video_filename}")

    # Process the reconstituted video with the YOLO model
    results = model(video_filename)
    #results[0].show()
    for result in results:
        result.show()
else:
    print("No frames received, video not created.")

# Optionally, export the model to ONNX format
#onnx_path = model.export(format="onnx")  # returns the path to the exported model
