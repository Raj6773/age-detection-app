# Age Detection App

This is a Flask-based web application for real-time age detection. It provides:
- Image upload option for age detection.
- Live video stream for real-time age detection using OpenCV.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/age-detection-app.git
   cd age-detection-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the app in a web browser at:
   ```
   http://localhost:5000
   ```

## Folder Structure

```
age-detection-app/
│-- 📄 app.py               # Main Flask application
│-- 📄 requirements.txt      # Required Python libraries
│-- 📄 README.md             # Project description
│-- 📂 templates/            # HTML templates
│   └── 📄 index.html        # Web interface
│-- 📂 static/               # Static files (CSS, JS, images)
```

## Deployment

To deploy on a cloud server:
- Use PythonAnywhere, Render, or Railway for free deployment.
- Configure `host='0.0.0.0'` in `app.py` for external access.

