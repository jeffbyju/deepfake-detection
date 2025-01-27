import os
import sys
import glob

from flask import Flask, request, jsonify, render_template
from utils import extract_frames, preprocess_frames
from model import DeepFakeDetector, load_model, predict
from absl import flags, app

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the deep fake detector model
model = load_model()

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_frames', 40, 'Number of Frames to extract from video')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = analyze_video(filepath)
        # Path to the directory containing frames
        frame_dir = os.path.join('static', 'uploads', f"{file.filename.split('.mp4')[0]}_frames")
        # Get list of frame file paths
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png'))[:FLAGS.num_frames])
        # Convert frame paths to URLs
        frame_urls = [f"/{path}" for path in frame_paths]
        return render_template('result.html', result=result, frames=frame_urls, num_frames=len(frame_urls))

def analyze_video(video_path):
    frames = extract_frames(video_path, )
    processed_frames = preprocess_frames(frames, video_path, FLAGS.num_frames, "./static/uploads")
    prediction, confidence = predict(model, processed_frames)
    return {
        "is_deepfake": prediction,
        "confidence": confidence
    }

if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(debug=True)