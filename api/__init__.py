# 3rd parth dependencies
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_httpauth import HTTPTokenAuth


db = SQLAlchemy()
Sec_key = 'the quick brown fox jumps over the lazy dog'
mail = Mail()
auth = HTTPTokenAuth(scheme='Bearer')


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
    global mail
    app = Flask(__name__)
    # mail = Mail(app)
    from api.routes import blueprint
    from api.routes_verify import blueprint1
    app.register_blueprint(blueprint)
    app.register_blueprint(blueprint1)
    
    app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USE_SSL'] = False
    app.config['MAIL_USERNAME'] = 'observah60@gmail.com'
    app.config['MAIL_PASSWORD'] = 'qkkvofrslzcalehp'
    app.config['MAIL_DEFAULT_SENDER'] = 'observah60@gmail.com'
    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:271202@localhost/Observa'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
    db.init_app(app)
    mail.init_app(app)
    # preload_deepface()
    app_context = app.app_context()
    app_context.push()
    db.create_all()
    app_context.pop()
    return app


