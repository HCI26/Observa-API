from api import db

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(32), unique=True, nullable=False, index=True)
    passhash = db.Column(db.String(64), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(13), unique=True, nullable=False)
    full_name = db.Column(db.String(64), nullable=False)
    date_of_birth = db.Column(db.Date)
    city = db.Column(db.String(64))
    country = db.Column(db.String(64), nullable=False)
    gender = db.Column(db.String(6))
    address = db.column(db.String(64))
    saved_visitors = db.relationship('SavedVisitor', back_populates='user', lazy=True)
    
    def __repr__(self):
        return '<User %r>' % self.username