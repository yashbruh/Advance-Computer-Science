from flask import Flask, request, jsonify, render_template
import os
import requests
from google.cloud import vision, videointelligence
from werkzeug.utils import secure_filename
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up Google Cloud credentials
service_account_file = "soy-analog-447121-p3-6f72d899b2c6.json"
if not os.path.exists(service_account_file):
    raise FileNotFoundError(f"Service account file '{service_account_file}' not found.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file

# Initialize Google Cloud clients
vision_client = vision.ImageAnnotatorClient()
video_client = videointelligence.VideoIntelligenceServiceClient()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

# Google API Key for Knowledge Graph
GOOGLE_KG_API_KEY = "AIzaSyBvazeKhGdpwiOTKs7jpyDwTOb4NjEqlCk"


def allowed_file(filename):
    """Check if file format is supported."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle file upload and process it based on file type."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if filename.lower().endswith(('jpg', 'jpeg', 'png', 'gif')):
            result = process_image(filepath)
        elif filename.lower().endswith(('mp4', 'mov', 'avi')):
            result = process_video(filepath)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    except Exception as e:
        result = {'error': f'Error processing file: {str(e)}'}

    os.remove(filepath)
    return jsonify(result)


def process_image(filepath):
    """Process an image file using Google Cloud Vision API."""
    with open(filepath, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    try:
        # Detect brands using logo detection
        logo_response = vision_client.logo_detection(image=image)
        brands = [logo.description for logo in logo_response.logo_annotations]

        if not brands:
            return {'error': 'No brands detected'}

        # Extract alcohol percentage for each brand from Google
        alcohol_data = {}
        for brand in brands:
            alcohol_info = fetch_alcohol_percentage_from_google(brand)
            alcohol_data[brand] = alcohol_info

    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}

    return {
        'type': 'image',
        'brands': brands,
        'alcohol_data': alcohol_data
    }


def process_video(filepath):
    """Process a video file using Google Video Intelligence API."""
    with open(filepath, 'rb') as video_file:
        input_content = video_file.read()

    try:
        # Request label detection from video
        operation = video_client.annotate_video(
            request={
                "features": [videointelligence.Feature.LOGO_RECOGNITION],
                "input_content": input_content,
            }
        )

        print("Processing video, please wait...")
        result = operation.result(timeout=300)

        brands = set()
        for annotation in result.annotation_results[0].logo_recognition_annotations:
            brands.add(annotation.entity.description)

        if not brands:
            return {'error': 'No brands detected in the video'}

        # Extract alcohol percentage for each brand
        alcohol_data = {}
        for brand in brands:
            alcohol_info = fetch_alcohol_percentage_from_google(brand)
            alcohol_data[brand] = alcohol_info

    except Exception as e:
        return {'error': f'Error processing video: {str(e)}'}

    return {
        'type': 'video',
        'brands': list(brands),
        'alcohol_data': alcohol_data
    }


def fetch_alcohol_percentage_from_google(brand):
    """Query Google Knowledge Graph API for alcohol percentage and type."""
    url = f"https://kgsearch.googleapis.com/v1/entities:search?query={brand}&key={GOOGLE_KG_API_KEY}&limit=1"

    try:
        response = requests.get(url)
        data = response.json()

        if "itemListElement" in data and data["itemListElement"]:
            for element in data["itemListElement"]:
                result = element.get("result", {})
                description = result.get("detailedDescription", {}).get("articleBody", "")
                alcohol_type = result.get("type", "Unknown")

                match = re.search(r'(\d{1,2}\.\d+|\d{1,2})%', description)
                alcohol_percentage = match.group(0) if match else "Unknown"

                return {
                    "percentage": alcohol_percentage,
                    "type": alcohol_type,
                    # "rating": assign_rating(alcohol_percentage),
                    # "description": description[:250] + "..." if description else "No description available"
                }

        return {"percentage": "Unknown", "type": "Unknown", "rating": "⭐⭐⭐", "description": "No description found"}

    except Exception as e:
        return {"percentage": "Unknown", "type": "Unknown", "rating": "⭐⭐⭐", "description": f"Error: {str(e)}"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
