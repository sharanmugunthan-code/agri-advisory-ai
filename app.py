"""
AI Agri-Advisory App - Python/Flask version
=============================================
This server does 3 things when a farmer submits a photo + crop + location:
  1. Classifies the photo using a model trained in Teachable Machine
  2. Fetches real weather data for the farmer's location
  3. Asks Gemini to turn that into a plain-language advisory

Run this with:  python app.py
Then open:      http://127.0.0.1:5000  in your browser
"""

from flask import Flask, render_template, request
import requests
import numpy as np
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# ============================================================
# 1. PUT YOUR KEYS HERE
# ============================================================
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY is not set")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
# ============================================================
# 2. LOAD YOUR TRAINED MODEL
#    (Export from Teachable Machine as "Tensorflow" -> "Keras" ->
#     download the .zip -> unzip -> you'll get keras_model.h5 and labels.txt
#     Put both files in the same folder as this app.py)
# ============================================================
from tensorflow.keras.models import load_model

# Load the trained agriculture disease model
model = load_model(
    "agri_model.keras",
    compile=False
)

print("Agriculture model loaded successfully!") 

# Read the class names (e.g. "Healthy", "Blight") from labels.txt
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]


# ============================================================
# STEP A: Classify the uploaded photo
# ============================================================
def classify_image(image_file):
    # Teachable Machine models expect a 224x224 image, normalized
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    img_array = np.asarray(image)
    normalized_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_array, axis=0)

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index]) * 100

    return f"{class_name} ({confidence:.1f}% confidence)"


# ============================================================
# STEP B: Get real weather for the farmer's location
# ============================================================
def get_weather(location):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Could not fetch weather. Check location spelling or API key.")

    data = response.json()
    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"]
    }


# ============================================================
# STEP C: Ask Gemini for a plain-language advisory
# ============================================================
def get_advisory(crop, location, weather, disease_prediction):
    prompt = f"""
You are an agricultural advisor helping a small farmer in India.

Crop: {crop}
Location: {location}
Weather: {weather['temp']}°C, {weather['humidity']}% humidity, {weather['condition']}
Photo diagnosis: {disease_prediction}

Give a short, practical, plain-language recommendation covering:
1. What the diagnosis means for this crop
2. One or two immediate actions the farmer should take
3. Any weather-related precaution

Keep it simple and avoid unnecessary technical jargon.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# ============================================================
# ROUTES: what happens when the browser talks to this server
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    crop = request.form.get("crop")
    location = request.form.get("location")
    photo = request.files.get("photo")

    if not crop or not location or not photo:
        return render_template("index.html", error="Please fill in all fields.")

    try:
        disease_prediction = classify_image(photo)
        weather = get_weather(location)
        advisory = get_advisory(crop, location, weather, disease_prediction)

        result = {
            "diagnosis": disease_prediction,
            "advisory": advisory
        }
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)