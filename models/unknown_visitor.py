from api import db

class UnknownVisitor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    image_path = db.Column(db.String)  # Store the image path
    embedding = db.Column(db.ARRAY(db.Float), nullable=False)

    def __repr__(self):
        return '<Unknown Visitor %r>' % self.id