from api import db
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import time
from itsdangerous import TimedSerializer as Serializer

Sec_key = 'the quick brown fox jumps over the lazy dog'

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(32), unique=True, nullable=False, index=True)
    passhash = db.Column(db.String(256), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(13), unique=True, nullable=False)
    full_name = db.Column(db.String(64), nullable=False)
    date_of_birth = db.Column(db.Date)
    city = db.Column(db.String(64))
    country = db.Column(db.String(64), nullable=False)
    gender = db.Column(db.String(6))
    address = db.column(db.String(64))
    saved_visitors = db.relationship('SavedVisitor', back_populates='user', lazy=True)
    #TODO AUTHENTICATION
    
    def hash_password(self, password):
        self.passhash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.passhash, password)
    
    
    def generate_auth_token(self, expires_in=6000):
        return jwt.encode({'id': self.id, 'exp': time.time() + expires_in}, Sec_key, algorithm='HS256')

    @staticmethod
    def verify_auth_token(token):
        try:
            data = jwt.decode(token, Sec_key, algorithms=['HS256'])
        except:
            return
        return User.query.get(data['id'])

    #TODO OAuthentiaction
    
    #TODO Reset Password
    
    #TODO SendMails
        
    def __repr__(self):
        return '<User %r>' % self.username