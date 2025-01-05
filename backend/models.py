from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ImageMetaData(db.Model):
    __tablename__ = 'image_metadata'
    
    id = db.Column(db.Integer, primary_key=True)  # Unique ID for the image
    url = db.Column(db.String(500), nullable=False)  # Cloudinary URL
    filename = db.Column(db.String(255), nullable=False)  # File name of the image
    tags = db.Column(db.ARRAY(db.String), nullable=True)  # Array of tags
    captions = db.Column(db.ARRAY(db.String), nullable=True)  # Array of captions
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Upload timestamp

    def __repr__(self):
        return f"<ImageMetadata(id={self.id}, filename={self.filename})>"
