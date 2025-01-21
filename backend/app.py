from flask import Flask,request, send_file, jsonify
import requests
from flask_cors import CORS
from PIL import Image
from models import db, ImageMetaData
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@db:5432/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
DETR_API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-101"
GPT2_IMAGE_CAPTIONING_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
SENTENCE_SIMILARITY = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": "Bearer " + os.getenv("HUGGINGFACE_API_KEY")}


db.init_app(app)

with app.app_context():
    db.create_all()  # This creates all tables defined by models
    
# Configuration  cloudinary     
cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"), 
    secure=True
)

# test route
@app.route('/')
def working():
    return "Flask backend working!"

#api to get image tags
@app.route('/generateTags', methods=['POST'])
def get_tags():
    print(request.files)
    print('Request received')
    if 'images' not in request.files:
        return jsonify({"error": "No images found in request"}), 400
    print('Images found in request')
    image_files = request.files.getlist('images')

    image_responses = []

    for image_file in image_files:
        data = image_file.read()
        gpt2_response = query_gpt2_image_captioning(data)

        detr_labels = []
        try:
            detr_response = query_detr_model(data)
            detr_labels = set(obj['label'] for obj in detr_response)
        except TypeError as e:
            # Handle the TypeError here, you can log it or return an error response
            error_message = f"Error processing image {image_file.filename}: {str(e)}"
            print(error_message)
        
        image_responses.append({
            "filename": image_file.filename,
            "tags": list(detr_labels),
            "captions": gpt2_response,
            "image_size":len(data)
        })

    print(image_responses)
    return {"error": False, "data": image_responses}, 200



def query_detr_model(data):
    response = requests.post(DETR_API_URL, headers=HEADERS, data=data, json={"parameters": {"wait_for_model": True}})
    return response.json()

def query_gpt2_image_captioning(data):
    response = requests.post(GPT2_IMAGE_CAPTIONING_API_URL, headers=HEADERS, data=data, json={"parameters": {"wait_for_model": True}})
    return response.json()[0]['generated_text']

# API to upload images and metadata
@app.route('/saveImage', methods=['POST'])
def save_image():
    if 'image' not in request.files or 'filename' not in request.form or 'tags' not in request.form or 'captions' not in request.form:
        return jsonify({"error": "Image, filename, tags, or captions missing in request"}), 400

    image_file = request.files['image']
    filename = request.form['filename']
    tags = request.form['tags'].split(',')  # Assuming tags are sent as a comma-separated string
    captions = request.form['captions'].split(',')  # Assuming captions are sent as a comma-separated string

    try:
        # Upload image to Cloudinary
        upload_response = cloudinary.uploader.upload(image_file, folder="uploaded_images")

        # Extract Cloudinary URL
        cloudinary_url = upload_response['secure_url']

        # Save metadata to the database
        image_metadata = ImageMetaData(
            url=cloudinary_url,
            filename=filename,
            tags=tags,
            captions=captions
        )
        db.session.add(image_metadata)
        db.session.commit()

        return jsonify({
            "error": False,
            "message": "Image uploaded and metadata saved successfully",
            "data": {
                "url": cloudinary_url,
                "filename": filename,
                "tags": tags,
                "captions": captions
            }
        }), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": True, "message": "Database error", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": True, "message": "Image upload failed", "details": str(e)}), 500

# API to get all images and metadata
@app.route('/getAllImages', methods=['GET'])
def get_all_images():
    try:
        images = ImageMetaData.query.all()
        image_data = []
        for image in images:
            image_data.append({
                "id": image.id,
                "url": image.url,
                "filename": image.filename,
                "tags": image.tags,
                "captions": image.captions,
                "timestamp": image.timestamp
            })
        return jsonify({"error": False, "data": image_data}), 200
    except SQLAlchemyError as e:
        return jsonify({"error": True, "message": "Database error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)