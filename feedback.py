from ultralytics import YOLO
import socket
import cv2
import numpy as np
import json

# Load the YOLO model
model = YOLO("yolo11n.pt")

# TCP server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(1)
print("Server listening on port 5000...")

conn, addr = server_socket.accept()
print(f"Connection established with {addr}")

# Video writer initialization variables
out = None
fps = 30  # Adjust as needed
video_filename = 'received_video.mp4'


def initialize_writer(frame_shape):
    """Initializes the video writer with the frame shape."""
    global out
    height, width, _ = frame_shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))


try:
    while True:
        # Receive the frame size (4 bytes)
        data = conn.recv(4)
        if not data:
            break
        size = int.from_bytes(data, 'little')

        # Receive the frame data
        frame_data = b''
        while len(frame_data) < size:
            packet = conn.recv(size - len(frame_data))
            if not packet:
                break
            frame_data += packet

        # Decode the frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Initialize video writer if not set
        if out is None:
            initialize_writer(frame.shape)

        # Save frame to video
        out.write(frame)

        # Real-time YOLO inference
        results = model(frame)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.tolist() if result.boxes is not None else []
            scores = result.boxes.conf.tolist() if result.boxes is not None else []
            classes = result.boxes.cls.tolist() if result.boxes is not None else []
            detections.append({"boxes": boxes, "scores": scores, "classes": classes})
            #result.show()  # Display detections

        # Send detection results back to client
        detection_data = json.dumps(detections).encode('utf-8')
        conn.send(len(detection_data).to_bytes(4, 'little'))  # Send length of data
        conn.send(detection_data)  # Send actual data

except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up
    conn.close()
    server_socket.close()
    if out:
        out.release()
    print(f"Video saved as {video_filename}")
