
from flask import Blueprint, abort, request, Response, stream_with_context
# import api.service as service

import cv2
from models import User,SavedVisitor
from datetime import datetime
# from main import app
import api.fake as fake
from api import db
from api.face_recognition import FaceRecognizer,FaceAnalyzer,VideoAnalyzer

blueprint = Blueprint("routes", __name__)



@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"

@blueprint.route("/gen_fake_users/<int:count>", methods=["POST"])
def generate_fake_users(count):
    fake.users(count)
    return "<h1>Generated {} fake users</h1>".format(count), 201


from flask import request
from deepface import DeepFace
from api import db
from models import SavedVisitor
import os

@blueprint.route("/add_visitor", methods=["POST"])
def add_visitor():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    name = input_args.get("name")
    relationship = input_args.get("relationship")
    user_id = input_args.get("user_id")
    img_url = input_args.get("img_url")
    
    

    if not all([name, relationship, user_id, img_url]):
        return {"message": "you must pass name, relationship, user_id, and img_url inputs"}

    #check if img path exists
    if not os.path.isfile(img_url):
        abort(Response("img_path does not exist", 400))
    
    result = DeepFace.represent(img_path=img_url,
                                   model_name="VGG-Face",
                                   detector_backend="retinaface",
                                   enforce_detection=True,
                                   align=True)
    
    if len(result) == 0:
        abort(Response("no face detected in image", 400))

    # print(result)
    embedding = result[0]["embedding"]
    new_visitor = SavedVisitor(
        name=name,
        relationship=relationship,
        user_id=user_id,
        embedding=embedding
    )
    db.session.add(new_visitor)
    db.session.commit()

    return {"message": "new visitor added successfully"}, 201



# Initialize dependencies
face_recognizer = FaceRecognizer()
face_recognizer.load_model(model_name="VGG-Face")
# face_analyzer = FaceAnalyzer()
video_analyzer =VideoAnalyzer(face_recognizer)


def gen_frames(user_id):
    """
    Generator function that continuously yields analyzed frames.
    """
    try:
        # Start capture thread
        url = "http://192.168.1.5:8080/video"
        cap = cv2.VideoCapture(url)
        saved_visitors = SavedVisitor.query.filter_by(user_id=user_id)
        
        try:
            while True:
                success, frame = cap.read()
                if success:
                    analyzed_frame, _, _ = video_analyzer.analyze_video(
                                            frame,
                                            saved_visitors,
                                            detector_backend="opencv")
                    # Encode frame and yield response
                    ret, buffer = cv2.imencode('.jpg', analyzed_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    break
        except Exception as e:
            print(f"Error in video capture_frames: {e}")

        
    except Exception as e:
        print(f"Error in video gen_frames: {e}")
    finally:
        # Stop capture thread and release resources
        cap.release()
        cv2.destroyAllWindows()



@blueprint.route('/video/<int:user_id>')
def video_feed(user_id):
    """
    Route that serves the video feed.
    """
    return Response(stream_with_context(gen_frames(user_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')