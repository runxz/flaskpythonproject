from flask import Flask, Response, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize variables for push-up counting
counter = 0
rep_started = False

# Define the push-up threshold (the y-coordinate of the hand)
push_up_threshold = 200  # Adjust this value based on your setup

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Endpoint for receiving video feed from the Flutter app
@app.route('/send_video', methods=['POST'])
def send_video():
    global counter, rep_started

    # Use request.data to get the video frame from the Flutter app
    frame = np.frombuffer(request.data, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Process the frame to detect the pose using MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        # Get the y-coordinate of the hand landmark (landmark 11)
        hand_y = int(landmarks[11].y * frame.shape[0])

        # Check if a push-up has started
        if not rep_started and hand_y > push_up_threshold:
            rep_started = True
        # Check if a push-up has been completed
        elif rep_started and hand_y < push_up_threshold:
            counter += 1
            rep_started = False

    return "OK"

# Endpoint for getting push-up count
@app.route('/get_pushup_count', methods=['GET'])
def get_pushup_count():
    return str(counter)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
