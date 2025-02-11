from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load Pre-trained Models
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe("models/deploy_age.prototxt", "models/age_net.caffemodel")

# Age Categories
AGE_GROUPS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Function to Detect Face & Age
def detect_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Prepare image for model
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.895), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_GROUPS[age_preds[0].argmax()]
        
        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, age, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

@app.route("/")
def index():
    return render_template("index.html")

# Live Video Streaming Route
def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_age(frame)
        _, buffer = cv2.imencode(".jpg", frame)
       def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_age(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        
        # ✅ FIXED `yield` statement
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Image Upload API
@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    image = np.asarray(bytearray(file.read()), dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Process image
    result_frame = detect_age(frame)
    _, buffer = cv2.imencode(".jpg", result_frame)
    
    return Response(buffer.tobytes(), content_type="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load Pre-trained Models
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe("models/deploy_age.prototxt", "models/age_net.caffemodel")

# Age Categories
AGE_GROUPS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Function to Detect Face & Age
def detect_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Prepare image for model
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.895), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_GROUPS[age_preds[0].argmax()]
        
        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, age, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

@app.route("/")
def index():
    return render_template("index.html")

# Live Video Streaming Route
def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_age(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Image Upload API
@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    image = np.asarray(bytearray(file.read()), dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Process image
    result_frame = detect_age(frame)
    _, buffer = cv2.imencode(".jpg", result_frame)
    
    return Response(buffer.tobytes(), content_type="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
