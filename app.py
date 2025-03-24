from ultralytics import YOLO

# server
import socket
import cv2
import numpy as np

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=10,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# TCP server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(0)
conn, addr = server_socket.accept()

while True:
    # Reception of the size of the incoming frame (4 bytes for an integer)
    data = conn.recv(4)
    if not data:
        break
    print("Server running")
    size = int.from_bytes(data, 'little')

    # Reception of the frame data based on the size
    frame_data = b''
    while len(frame_data) < size:
        packet = conn.recv(size - len(frame_data))
        if not packet:
            break
        frame_data += packet

    # Decoding the frame from JPEG format
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame)
    results[0].show()





# Perform object detection on an image
# results = model("family.jpg")
# results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model