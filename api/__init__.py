# 3rd parth dependencies
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

def preload_deepface():
    from deepface import DeepFace
    # from deepface.detectors import FaceDetector
    # DeepFace.build_model('VGG-Face')
    # FaceDetector.build_model("retinaface")
    DeepFace.represent(img_path="./dataset/bezos.jpg",
                                   model_name="VGG-Face",
                                   detector_backend="retinaface",
                                   enforce_detection=True,
                                   align=True)

def create_app():
    app = Flask(__name__)
    
    from api.routes import blueprint
    app.register_blueprint(blueprint)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres@localhost/Observa'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    # preload_deepface()
    app_context = app.app_context()
    app_context.push()
    db.create_all()
    app_context.pop()
    return app


