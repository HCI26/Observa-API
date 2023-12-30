from flask import Blueprint, abort, request, Response, stream_with_context, Flask, g, send_file
# import api.service as service
from flask_httpauth import HTTPTokenAuth

import cv2
from models import User,SavedVisitor,Visit
from datetime import datetime
# from main import app
import api.fake as fake
from api import db
from api.face_recog import FaceRecognizer,FaceAnalyzer,VideoAnalyzer,VideoAnalysisOutput
from datetime import date
from flask import request, jsonify
from deepface import DeepFace
from api import db, auth
from models import SavedVisitor
import os, json
from api.UserDTO import UserDTO
from api.VisitorDTO import VisitorDTO
from flask_cors import CORS
import base64
import numpy as np
import uuid

blueprint = Blueprint("routes", __name__)
CORS(blueprint)
'''End point that lists the current user's information'''

@blueprint.route("/api/user/info", methods=["GET"])
@auth.login_required
def get_user_info():
    # user_id = 1 # TODO: change to cookie
    # user = User.query.get(user_id)
    user = g.user
    if user is None:
        abort(404, description="User not found") # return 404 not found

    user_dto = UserDTO(
        username=user.username,
        fullname=user.full_name,
        email=user.email,
        phonenumber=user.phone_number,
        dateofbirth=user.date_of_birth,
        country=user.country,
        city=user.city,
        gender=user.gender,
        address=user.address
    )
    print(user_dto)
    return jsonify(user_dto.__dict__)

'''End point that lists the current user's visitors'''

@blueprint.route("/api/user/visitors/get", methods=["GET"])
@auth.login_required
def list_users_of_visitors():
    # user_id = 1 # TODO: change to cookie
    saved_visitors = SavedVisitor.query.filter_by(user_id=g.user.id).all()
    if not saved_visitors:
        return jsonify("[]")
    visitors_dtos=[
        VisitorDTO(
            id=visitor.id,
            name=visitor.name,
            relation=visitor.relationship,
            date=12312412412,            # TODO: Change date to epochtime in database
            img_path= visitor.image_path
        )
        for visitor in saved_visitors
    ]
    return jsonify(visitors=[visitor_dto.__dict__ for visitor_dto in visitors_dtos])

def save_image(file):
    # Save the image to a specific directory or process it as needed
    # For example, save the file to the 'uploads' folder
    file.save('dataset/' + file.filename)
    return 'dataset/' + file.filename


@blueprint.route('/uploads/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_file(f'{os.getcwd()}/uploads/{filename}', as_attachment=True)
    except FileNotFoundError:
        abort(404)

@blueprint.route('/api/user/visitors/add', methods=['POST'])
@auth.login_required
def add_visitor():
    # Assuming the VisitorDTO is sent as JSON data
    userID = g.user.id 
    name = request.form.get('name')
    relation = request.form.get('relation')
    if 'image' not in request.files:
        abort(401)
    image_file = request.files["image"]
    extension = image_file.filename.split('.')[1]
    path = "uploads/" + str(uuid.uuid4()) + "." + extension
    if image_file:
        image_file.save(path)

    try:
        result = DeepFace.represent(img_path=path,
                                    model_name="VGG-Face",
                                    detector_backend="retinaface",
                                    enforce_detection=True,
                                    align=True)
    except:
        abort(Response("no face detected in image", 400))

    if len(result) == 0:
        abort(Response("no face detected in image", 400))

    embedding = result[0]["embedding"]
    new_visitor = SavedVisitor(
        name=name,
        relationship=relation,
        user_id=userID,
        embedding = embedding,
        image_path=path        
    )
    db.session.add(new_visitor)
    db.session.commit()


    return "Visitor created successfully", 201


@blueprint.route('/api/user/visitors/edit', methods=['POST'])
@auth.login_required
def edit_visitor():
    userID = g.user.id

    name = request.form.get('name')
    relation = request.form.get('relation')
    id = request.form.get('id')
    if 'image' not in request.files:
        abort(401)
    image_file = request.files["image"]
    extension = image_file.filename.split('.')[1]
    path = "uploads/" + str(uuid.uuid4()) + "." + extension
    if image_file:
        image_file.save(path)

    try:
        result = DeepFace.represent(img_path=path,
                                    model_name="VGG-Face",
                                    detector_backend="retinaface",
                                    enforce_detection=True,
                                    align=True)
    except:
        abort(Response("no face detected in image", 400))

    if len(result) == 0:
        abort(Response("no face detected in image", 400))

    embedding = result[0]["embedding"]
    new_visitor = SavedVisitor(
        name=name,
        relationship=relation,
        user_id=userID,
        id=id,
        embedding = embedding,
        img_path=path
    )
    visitor = SavedVisitor.query.filter_by(id=id).first()
    db.session.delete(visitor)
    db.session.commit()
    db.session.add(new_visitor)
    db.session.commit()


    # return jsonify(image_path)
    return "Visitor edited successfully", 201


@blueprint.route('/user/visitors/delete/<int:id>', methods=['DELETE'])
@auth.login_required
def delete_visitor(id):

    visitor = SavedVisitor.query.filter_by(id=id).first()
    db.session.delete(visitor)
    db.session.commit()

    return{"message": "Visitor deleted successfully"}, 201

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' # My edits is before this line
@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"

@blueprint.route("/gen_fake_users/<int:count>", methods=["POST"])
def generate_fake_users(count):
    fake.users(count)
    return "<h1>Generated {} fake users</h1>".format(count), 201


@blueprint.route("/change_data", methods=["GET","POST"])
@auth.verify_token
def change_data():
    row_to_update = g.user

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


@blueprint.route("/settings", methods=["GET"])
@auth.verify_token
def settings():
    user_data = g.user
    
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

    visitors = SavedVisitor.query.filter_by(id=id).all()

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


@blueprint.route("/api/history", methods=["GET"])
def get_history():
    history = Visit.query.filter_by(user_id=g.user.id).all()
    # history_out = [
    #     {
    #         'id': visitor.id,
    #         'name': visitor.name,
    #         'relationship': visitor.relationship,
    #         'embedding': visitor.embedding,
    #         'last_visited': visitor.last_visited
    #     }
    #     for visit in history
    # ]

@blueprint.route("/add_visitor", methods=["POST"])
def addvisitor():
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

    url = "http://192.168.1.17:8080/video"
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        saved_visitors = SavedVisitor.get_saved_visitors(user_id)
        unknown_visitors = SavedVisitor.get_unknown_visitors(user_id)
        if success:
            output:VideoAnalysisOutput = video_analyzer.analyze_video(
                                    frame,
                                    current_user_id=user_id,
                                    saved_visitors=saved_visitors,
                                    unknown_visitors=unknown_visitors,
                                    detector_backend="opencv",
                                    model_name="VGG-Face",
                                    enable_face_analysis=False)
            
            # Encode frame and yield response
            ret, buffer = cv2.imencode('.jpg', output.drawn_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



@blueprint.route('/video/<int:user_id>')
def video_feed(user_id):
    """
    Route that serves the video feed.
    """
    return Response(stream_with_context(gen_frames(user_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')