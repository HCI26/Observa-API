from flask import Blueprint, abort, jsonify, request, g, Response, stream_with_context, url_for
from flask_httpauth import HTTPTokenAuth
from jwt import decode
from flask_cors import CORS
from flask_mail import Message, Mail
import cv2
from models import User,SavedVisitor
from datetime import datetime
import api.fake as fake
from api import db
from api import mail,auth

blueprint1 = Blueprint("routes_verify", __name__)
CORS(blueprint1)


@auth.verify_token
def verify_password(token):
    user = User.verify_auth_token(token)
    if not user:
        return False
    g.user = user
    return True

@blueprint1.route('/api/test_token')
@auth.login_required
def test_token():
    return jsonify({'username': g.user.username})
    
@blueprint1.route('/api/users/signup', methods=["POST"])
def new_user():
    username = request.json.get('username')
    passhash = request.json.get('password')
    email = request.json.get('email')
    phone_number = request.json.get('phone_number')
    full_name = request.json.get('full_name')
    date_of_birth = request.json.get('date_of_birth')
    city = request.json.get('city')
    country = request.json.get('country')
    gender = request.json.get('gender')
    address = request.json.get('address')
    if username is None or passhash is None:
        abort(400)
    if User.query.filter_by(username = username).first() is not None or User.query.filter_by(email = email).first() is not None:
        abort(400)
    
    user = User(username = username, email=email, phone_number = phone_number, full_name = full_name,
                date_of_birth = date_of_birth, city= city, country = country, gender = gender, address = address)
    user.hash_password(passhash)
    db.session.add(user)
    db.session.commit()
    token = user.generate_auth_token(6000)
    return jsonify({'username': user.username,'token': token})


@blueprint1.route('/api/users/login', methods=['POST'])
def log_in():
    username = request.json.get('username')
    passhash = request.json.get('password')
    user = User.query.filter_by(username = username).first()
    if not user or not user.verify_password(passhash):
        abort(401)
    
    token = user.generate_auth_token(6000)
    return jsonify({'data': 'Hello, %s!' % user.username, 'token': token})

@blueprint1.route('/api/users/reset_password', methods=['POST'])
def reset():
    username = request.json.get('username')
    user = User.query.filter_by(username=username).first()
    if not user:
        abort(403)
    if not user.email:
        abort(401)
    generated_password = user.generate_random_password()
    msg = Message('Password Reset', sender='observah60@gmail.com', recipients=[user.email])
    msg.body = f"Hi, {user.username}\n\tYour generated password is "+generated_password+" \nChange this as soon as possible, for your security concerns.\nObserva Support"
    mail.send(msg)
    user.hash_password(generated_password)
    db.session.commit()
    return "Message Sent Successfully",201 

@blueprint1.route('/sendmail')
def send_mail():
    msg = Message('<Password Reset>', sender='observah60@gmail.com', recipients=['moustafaesam20113@gmail.com'])
    msg.body = "Hi, Moustafa"
    mail.send(msg)
    return "Message Sent Successfully",201