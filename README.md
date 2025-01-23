
# Intelligent Image Management Backend

This backend serves as the core for an **AI-powered Image Management System**, handling image uploads, tag generation, search functionalities, and database operations. It is integrated with **PostgreSQL** for metadata storage and **Cloudinary** for image hosting.

## Frontend Repository

[Frontend GitHub Repository](https://github.com/042Vidhi/Intelligent-Image-Management-System-Frontend)  

---
## Features  

- **Image Upload and Metadata Generation**: Automatically tags and captions images using HuggingFace models.  
- **Cloudinary Integration**: Stores uploaded images securely on Cloudinary.  
- **PostgreSQL Database**: Stores metadata for efficient search and retrieval.  
- **Search Functionality**: AI-powered similarity matching for tags and captions.  
- **Dockerized Setup**: Simplified environment setup and deployment using Docker.  

---

## Tech Stack

- **Python** (Flask)  
- **PostgreSQL** (Database)
- **SQLAlchemy**  (ORM)
- **Cloudinary** (for image storage)  
- **Docker & Docker Compose**  (Containerization)
- **HuggingFace Transformers** (AI-powered tagging and captioning)  

---

## APIs and Their Functionality

### 1. **`/`** (GET)  
**Description**: Health check endpoint to verify the Flask backend is running. 
 
**Response**:
Flask backend working!

---

### 2. **`/generateTags`** (POST)  
**Description**: Generates tags and captions for uploaded images using AI models.  
**Request Body**:
- `images`: One or multiple image files (as `multipart/form-data`).  

**Screenshot**:


---

### 2. **`/saveImage`** (POST)  
**Description**: Uploads an image to Cloudinary and saves metadata in PostgreSQL.  
**Request Body**:
- `image`: Image file (as `multipart/form-data`).  
- `filename`: Image name.  
- `tags`: Comma-separated list of tags.  
- `captions`: Comma-separated list of captions.  

**Response**:
```json
{
    "error": false,
    "message": "Image uploaded and metadata saved successfully",
    "data": {
        "url": "https://cloudinary-link/image.jpg",
        "filename": "example.jpg",
        "tags": ["dog", "petname"],
        "captions": ["a dog is playing in the garden"]
    }
}

```
**Screenshot**:

---

### 3. **`/getAllImages`** (GET)  
**Description**: Retrieves all saved images and metadata.  

**Response**:
```json
{
    "error": false,
    "data": [
        {
            "id": 1,
            "url": "https://cloudinary-link/image.jpg",
            "filename": "example.jpg",
            "tags": ["dog", "pet"],
            "captions": ["a dog is playing in the garden"],
            "timestamp": "2025-01-23T10:30:00Z"
        }
    ]
}
```
**Screenshot**:
---

### 4. **`/search`** (GET)  
**Description**: Searches for images based on a query, matching tags or captions (case-insensitive).  

**Request Query Parameters**:  
- `query`: The search term.  

**Response**:
```json
{
    "error": false,
    "data": [
        {
            "id": 1,
            "url": "https://cloudinary-link/image.jpg",
            "filename": "example.jpg",
            "tags": ["dog", "pet"],
            "captions": ["a dog is playing in the garden"],
            "score": 0.95
        }
    ]
}
```
**Screenshot**:

---

## How to Run the Project  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/042Vidhi/Intelligent-Image-Management-System-Backend
   cd backend
   ```

2. **Set Up Environment Variables**  
   Create a `.env` file and add the following variables:  
   ```env
   CLOUDINARY_CLOUD_NAME=<your_cloudinary_cloud_name>
   CLOUDINARY_API_KEY=<your_cloudinary_api_key>
   CLOUDINARY_API_SECRET=<your_cloudinary_api_secret>
   HUGGINGFACE_API_KEY=<your_huggingface_api_key>
   ```

3. **Start the Application with Docker**  
   Ensure Docker is installed, then run:  
   ```bash
   docker-compose up --build
   ```

4. **Access the Backend**  
   The backend will run on `http://localhost:5000`.

---



