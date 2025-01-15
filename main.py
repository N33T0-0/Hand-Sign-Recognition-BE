from typing import Union

import base64
# import cv2
import numpy as np
import keras
import mediapipe as mp

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8501",
]

class Item(BaseModel):
    frame: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.post("/process-frame")
# def read_item(item:Item):
#     # Send a response if needed
#     encoded_data = item.split(',')[1]
#     return ({'status': 'frame received'})

def base64_to_image(base64_str):
    """Convert Base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_str)
    np_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def image_to_base64(image):
    """Convert OpenCV image to Base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Decode
target = ['a','b','c','e','i','m','o','s','t','u']

# Load Model in
classification_model = keras.models.load_model('nn.keras')
# classification_model = joblib.load('random_forest.joblib')

def processNClassify(X):
    pred = classification_model.predict(X)
    # print(target[np.argmax(pred)])
    # print(np.argmax(pred))
    return target[np.argmax(pred)]

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# def detect_handsign(frame):
#     # Convert the frame to RGB format
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     h, w, c = frame.shape
    
#     # Process the frame to detect hands
#     results = hands.process(frame_rgb)
    
#     # Process the frame to detect hands
#     results = hands.process(frame_rgb)

#     cropped_frame = frame
    
#     position = []
#     edges = []

#     # Check if hands are detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw Bounding Box
#             x_max = 0
#             y_max = 0
#             x_min = w
#             y_min = h
#             for lm in hand_landmarks.landmark:
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 position.append([x,y])
#                 if x > x_max:
#                     x_max = x + 30
#                 if x < x_min:
#                     x_min = x - 30
#                 if y > y_max:
#                     y_max = y + 30
#                 if y < y_min:
#                     y_min = y - 30
#             if x_min < 0:
#                 x_min = 0
#             if y_min < 0:
#                 y_min = 0
#             position_ary = np.asarray(position)
#             position_ary = position_ary.reshape(1,-1)

#             # # Draw landmarks on the frame
#             # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             cropped_frame = frame[y_min:y_max,x_min:x_max]
            
#             temp_img = cv2.resize(cropped_frame,(100,100))
#             temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

#             edges = cv2.Canny(temp_img,75,150)
#             edges = np.asarray(edges)
#             edges = edges.reshape(1,-1)

#             frame_norm = temp_img.reshape(1,-1)
#             frame_norm = frame_norm/255

#             img_zip = np.hstack((frame_norm,position_ary,edges))

            
#             if position_ary.shape[1] == 42:
#                 pred = processNClassify(img_zip)
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x_min, y_min - 20), (x_min + 20, y_min), (0,255,0), -1)
#                 cv2.putText(frame, pred, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    
#         # return {"target":pred, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
#         return frame
#     return frame


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         # Receive JSON data containing the frame from the frontend
#         data = await websocket.receive_text()
#         frame_json = json.loads(data)
#         base64_frame = frame_json['frame']

#         # Decode the frame
#         frame = base64_to_image(base64_frame)

#         # Process the frame 
#         # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         processed_img = detect_handsign(frame)

#         if processed_img is not None:
#             # Encode the processed frame back to Base64
#             processed_base64 = image_to_base64(processed_img)

#             # Send the processed frame back to the frontend
#             await websocket.send_text(json.dumps({"frame": processed_base64}))

#         # # Process frame (get bounding box coords + target)
#         # json_string = detect_handsign(frame)

#         # if json_string is not None:
#         #     await websocket.send_text(json.dumps(json_string))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive JSON data containing the frame from the frontend
        data = await websocket.receive_text()
        frame_json = json.loads(data)
        base64_frame = frame_json['frame']
        # Send the processed frame back to the frontend
        await websocket.send_text(json.dumps({"frame": base64_frame}))

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         # Receive the frame from the frontend
#         frame_bytes = await websocket.receive_bytes()

#         # Decode the image from bytes
#         nparr = np.frombuffer(frame_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Process the frame (e.g., add bounding box or grayscale)
#         h, w, _ = frame.shape
#         cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 4)  # Add bounding box
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#         processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for display

#         # Encode the processed frame as JPEG
#         _, buffer = cv2.imencode('.jpg', processed_frame)

#         # Send the processed frame back to the frontend
#         await websocket.send_bytes(buffer.tobytes())
