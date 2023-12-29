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


from flask import request, jsonify
from deepface import DeepFace
from api import db
from models import SavedVisitor
import os, json

@blueprint.route("/change_data/<int:id>", methods=["GET","POST"])
def change_data(id):
    row_to_update = User.query.get(id)

    if request.method == 'POST':
        input_args = request.get_json()

        if input_args is None:
            return {"message": "empty input set passed"}

        if input_args.get("username") is not None:
            row_to_update.username = input_args.get("username")
        if input_args.get("passhash") is not None:
            row_to_update.passhash = input_args.get("passhash")
        if input_args.get("email") is not None:
            row_to_update.email = input_args.get("email")
        if input_args.get("number") is not None:
            row_to_update.number = input_args.get("number")
        if input_args.get("full_name") is not None:
            row_to_update.full_name = input_args.get("full_name")
        if input_args.get("city") is not None:
            row_to_update.city = input_args.get("city")
        if input_args.get("address") is not None:
            row_to_update.address = input_args.get("address")            
        
        db.session.commit()
        return{"message": "User data updated successfully"}, 201
    return{"message": "failed to update user"}

@blueprint.route("/settings/<int:id>", methods=["GET"])
def settings(id):
    user_data = User.query.get(id)
    
    if user_data is None:
        return jsonify({"error": "failed to load user data"}), 404

    return json.dumps({"username": user_data.username,
            "passhash": user_data.passhash,
            "email": user_data.email,
            "phone_number": user_data.phone_number,
            "full_name": user_data.full_name,
            "date_of_birth": user_data.date_of_birth.strftime("%Y-%m-%d"),
            "city": user_data.city,
            "country": user_data.country,
            "gender": user_data.gender,
            #"address": user_data.address
            })

@blueprint.route("/history/<int:user_id>", methods=["GET"])
def get_visitors(user_id):
    user = User.query.get(user_id)

    if user is None:
        return jsonify({'error': 'User not found'}), 404

    visitors = SavedVisitor.query.filter_by(user_id=id).all()

    # Convert visitors to a list of dictionaries
    visitors_list = [
        {
            'id': visitor.id,
            'name': visitor.name,
            'relationship': visitor.relationship,
            'embedding': visitor.embedding,
            'last_visited': visitor.last_visited
        }
        for visitor in visitors
    ]

    return jsonify({'visitors': visitors_list})



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
    # try:
    # Start capture thread
    url = "http://192.168.1.17:8080/video"
    cap = cv2.VideoCapture(0)
    saved_visitors = SavedVisitor.query.filter_by(user_id=user_id)

    # try:
    while True:
        success, frame = cap.read()
        if success:
            analyzed_frame, _, _, _ = video_analyzer.analyze_video(
                                    frame,
                                    saved_visitors,
                                    detector_backend="retinaface",
                                    enable_face_analysis=False)
            # Encode frame and yield response
            ret, buffer = cv2.imencode('.jpg', analyzed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
        # except Exception as e:
        #     print(f"Error in video capture_frames: {e}")

        
    # except Exception as e:
    #     print(f"Error in video gen_frames: {e}")
    # finally:
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
