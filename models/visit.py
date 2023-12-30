from api import db

class Visit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    visitor_id = db.Column(db.Integer, db.ForeignKey("saved_visitors.id"))
    unknown_visitor_id = db.Column(db.Integer, db.ForeignKey("unknown_visitor.id"))  # Link to UnknownVisitor if applicable
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    timestamp = db.Column(db.Integer, nullable=False)

    visitor = db.relationship("SavedVisitor", backref="visits")
    unknown_visitor = db.relationship("UnknownVisitor", backref="visits")  # Optional relationship
    user = db.relationship("User", backref="visits")