
from flask import Blueprint, abort, request, Response, stream_with_context
import api.service as service
import cv2
from models import User,SavedVisitor
from datetime import datetime
# from main import app
import api.fake as fake
from api import db

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


def gen_frames(user_id):  
    url = "http://192.168.1.5:8080/video"
    cap = cv2.VideoCapture(url)
    # with app.app_context():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            continue
        else:
            analyzed_frame = service.videoAnalysis(frame,SavedVisitor.query.filter_by(user_id=user_id),detector_backend="opencv")
            ret, buffer = cv2.imencode('.jpg', frame)
            analyzed_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + analyzed_frame + b'\r\n')  # concat frame one by one and show result
            
@blueprint.route('/video/<int:user_id>')
def video_feed(user_id):
    return Response(stream_with_context(gen_frames(user_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')