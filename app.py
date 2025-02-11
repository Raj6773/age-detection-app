from flask import Flask, render_template, request, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

# Age categories
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def detect_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.895), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_GROUPS[age_preds[0].argmax()]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, age, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_age(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame
'
               b'Content-Type: image/jpeg

' + buffer.tobytes() + b'
')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
