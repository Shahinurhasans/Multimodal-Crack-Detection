# ===================================================================
# Concrete Crack Detection - Local Website (Flask backend)
# ===================================================================
# This is a small web server. It does three things:
#   1. Shows a web page with two upload boxes (image + audio)
#   2. When you click "Analyze", it receives the uploaded files,
#      prepares them EXACTLY like during training, and asks the
#      saved hybrid model for a prediction
#   3. Sends the result (Crack / No crack + probability) back to
#      the page, which displays it
#
# How to run this:
#   1. Open a terminal in this "webapp" folder
#   2. Install the needed packages (only once):
#        pip install flask tensorflow librosa numpy pillow
#   3. Start the server:
#        python app.py
#   4. Open your browser at:  http://127.0.0.1:5000
# ===================================================================

import csv
import os
from datetime import datetime

import numpy as np
import librosa
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

load_dotenv()

from report_generator import generate_report

# -------------------------------------------------------------
# Settings - these MUST match the values used in the training script,
# otherwise the model will receive data shaped differently than it
# learned from, and its predictions will not make sense.
# -------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
SPEC_HEIGHT = 128     # frequency bands in the audio spectrogram "image"
SPEC_WIDTH = 128      # time-steps in the audio spectrogram "image"
SAMPLE_RATE = 22050   # audio samples per second (librosa's default)

MODEL_PATH = r"F:\Crack Detection\crack_detection_hybrid_model.h5"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
REPORTS_CSV = os.path.join(os.path.dirname(__file__), "reports.csv")
CSV_FIELDS = ["timestamp", "image_file", "audio_file", "label", "probability", "report"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def log_report_to_csv(row):
    """Append one inspection record to reports.csv, writing the header first time."""
    file_exists = os.path.isfile(REPORTS_CSV)
    with open(REPORTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

app = Flask(__name__)

# The model is loaded ONCE, when the server starts - this way every
# upload is analyzed instantly, instead of reloading the model each time.
print("Loading the trained model ... please wait.")
model = load_model(MODEL_PATH)
print("Model loaded! The website is ready.")


# -------------------------------------------------------------
# Turn an uploaded image into the same format the model was trained on
# -------------------------------------------------------------
def prepare_image(file_path):
    img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0          # scale pixel values to 0-1
    return np.expand_dims(img_array, axis=0)        # add a "batch" dimension -> shape (1, H, W, 3)


# -------------------------------------------------------------
# Turn an uploaded audio clip into the same kind of spectrogram
# "image" the model was trained on
# -------------------------------------------------------------
def prepare_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=SPEC_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Make it exactly SPEC_WIDTH time-steps wide (cut or pad with silence)
    if spectrogram_db.shape[1] > SPEC_WIDTH:
        spectrogram_db = spectrogram_db[:, :SPEC_WIDTH]
    else:
        pad_amount = SPEC_WIDTH - spectrogram_db.shape[1]
        spectrogram_db = np.pad(
            spectrogram_db,
            pad_width=((0, 0), (0, pad_amount)),
            mode="constant",
            constant_values=spectrogram_db.min()
        )

    # Scale to 0-1, the same way the training script did
    spectrogram_db = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
    spectrogram_db = spectrogram_db[..., np.newaxis]   # add the "channel" dimension -> (H, W, 1)
    return np.expand_dims(spectrogram_db, axis=0)       # add a "batch" dimension -> (1, H, W, 1)


# -------------------------------------------------------------
# Page routes
# -------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    audio_file = request.files.get("audio")

    if image_file is None or image_file.filename == "" or audio_file is None or audio_file.filename == "":
        return jsonify({"error": "Please choose both an image file and an audio file."}), 400

    # Save the uploaded files temporarily so librosa/Keras can read them from disk
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    image_file.save(image_path)
    audio_file.save(audio_path)

    image_input = prepare_image(image_path)
    audio_input = prepare_audio(audio_path)

    # The model outputs one number between 0 and 1:
    #   close to 1 -> crack,   close to 0 -> no crack
    probability = float(model.predict([image_input, audio_input])[0][0])
    label = "Crack detected" if probability >= 0.5 else "No crack detected"

    # Ask the LLM (grounded with retrieved knowledge-base notes -- a small
    # RAG pipeline) to turn the raw verdict into a readable inspection report.
    report = generate_report(image_path, label, probability)

    log_report_to_csv({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_file": image_file.filename,
        "audio_file": audio_file.filename,
        "label": label,
        "probability": round(probability, 4),
        "report": report,
    })

    return jsonify({
        "label": label,
        "probability": round(probability, 4),
        "report": report
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
