import io
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import base64
# from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# hf_client = InferenceClient(
#     provider="hf-inference",
#     api_key=os.getenv("HUGGINGFACE_API_KEY")
# )

# nebius_client = InferenceClient(
#     provider="nebius",
#     api_key=os.getenv("HUGGINGFACE_API_KEY")
# )

# GPT2_IMAGE_CAPTIONING_MODEL = "nlpconnect/vit-gpt2-image-captioning"
# DETR_MODEL="facebook/detr-resnet-101"
# LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@db:5432/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
DETR_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-101"
LLAVA_API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
# GPT2_IMAGE_CAPTIONING_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
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
    logging.warning('Images found in request')
    image_files = request.files.getlist('images')

    image_responses = []

    for image_file in image_files:
        data = image_file.read()
        try:
            gpt2_response = query_llava_model_image_captioning_api(data)
        except TypeError as e:
            # Handle the TypeError here, you can log it or return an error response
            error_message = f"Error processing image {image_file.filename}: {str(e)}"
            logging.warning(error_message)
            gpt2_response = ""

        detr_labels = []
        try:
            detr_response = query_detr_model_api(data)
            logging.warning(f"Result from DETR model: {detr_response}")
            detr_labels = set(detr_response)
        except TypeError as e:
            # Handle the TypeError here, you can log it or return an error response
            error_message = f"Error processing image {image_file.filename}: {str(e)}"
            logging.warning(error_message)
        
        image_responses.append({
            "filename": image_file.filename,
            "tags": list(detr_labels),
            "captions": gpt2_response,
            "image_size":len(data)
        })

    print(image_responses)
    return {"error": False, "data": image_responses}, 200


#using lalava model with api
def query_llava_model_image_captioning_api(data):

    try:
        encoded_data = base64.b64encode(data).decode("utf-8")
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_data}"
                            }
                        }
                    ]
                }
            ],
            "model": "llava-hf/llava-1.5-7b-hf"
        }
        response = requests.post(LLAVA_API_URL, headers=HEADERS, json=payload)
        message = ""
        logging.warning(f"Response from LLAVA model API: {response}")
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
        else:
            logging.warning(f"LLAVA model API error: {response.status_code}")
    except Exception as e:
        logging.warning(f"Error querying LLAVA model API: {str(e)}")
        message = ""

    logging.warning(f"Response from LLAVA model API: {message}")
    return message



# detr model with api
def query_detr_model_api(data):
    try:
        response = requests.post(DETR_API_URL, headers={"Content-Type": "image/jpeg",**HEADERS}, data=data)
        logging.warning(f"Response from DETR model aPI: {response}")
        if response.status_code == 200:
            result = response.json()
            logging.warning(f"Result from DETR model: {result}")
            labels = [obj['label'] for obj in result]
        else:
            logging.warning(f"DETR model API error: {response.status_code}, {response.text}")
            labels = []
    except Exception as e:
        logging.warning(f"Error querying DETR model API: {str(e)}")
        labels = []

    logging.warning(f"Response from DETR model API: {labels}")
    return labels


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


def embed_text(text):
    response = requests.post(
        SENTENCE_SIMILARITY,
        headers=HEADERS,
        json={"inputs": text}
    )
    response_data = response.json()
    if isinstance(response_data, list):
        return np.array(response_data[0])  # Return the embedding as a numpy array
    else:
        raise ValueError(f"Failed to embed text: {response_data}")

#API to search images
@app.route('/search', methods=['GET'])
def search_images():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": True, "message": "Query parameter is required"}), 400

    query = query.lower()  # Normalize query for case-insensitive matching

    try:
        images = ImageMetaData.query.all()  # Fetch all images from the database
        matching_images = []
        seen_ids = set()  # To track unique image IDs

        for image in images:
            for tag_or_caption in image.tags + image.captions:
                tag_or_caption_lower = tag_or_caption.lower()

                # 1. Check for manual substring match
                if query in tag_or_caption_lower:
                    if image.id not in seen_ids:
                        matching_images.append({
                            "id": image.id,
                            "url": image.url,
                            "filename": image.filename,
                            "tags": image.tags,
                            "captions": image.captions,
                            "score": 1.0  # Maximum score for exact or substring match
                        })
                        seen_ids.add(image.id)  # Mark this image ID as seen
                    continue  # Skip similarity check for this tag/caption

                # 2. Use similarity search
                payload = {"inputs": [query, tag_or_caption]}
                response = requests.post(SENTENCE_SIMILARITY, headers=HEADERS, json=payload)

                if response.status_code == 200:
                    try:
                        similarity_score = response.json()[0]['score']
                        if similarity_score > 0.5:  # Adjust threshold as needed
                            if image.id not in seen_ids:
                                matching_images.append({
                                    "id": image.id,
                                    "url": image.url,
                                    "filename": image.filename,
                                    "tags": image.tags,
                                    "captions": image.captions,
                                    "score": similarity_score
                                })
                                seen_ids.add(image.id)  # Mark this image ID as seen
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing similarity API response: {e}")
                else:
                    print(f"Error from similarity API: {response.status_code}, {response.text}")

        # Sort matching images by similarity score in descending order
        matching_images = sorted(matching_images, key=lambda x: x['score'], reverse=True)

        return jsonify({"error": False, "data": matching_images}), 200

    except SQLAlchemyError as e:
        return jsonify({"error": True, "message": "Database error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
