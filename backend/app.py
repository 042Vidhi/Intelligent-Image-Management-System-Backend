from flask import Flask,request, send_file, jsonify
import requests
from flask_cors import CORS
from PIL import Image
from models import db, ImageMetaData
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

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



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)