from api import db

class SavedVisitor(db.Model):
    __tablename__ = 'saved_visitors'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(64), nullable=False)
    relationship = db.Column(db.String(64))
    embedding = db.Column(db.ARRAY(db.Float), nullable=False)
    last_visited = db.Column(db.DateTime)
    user = db.relationship('User', back_populates='saved_visitors', lazy=True)
    
    def __repr__(self):
        return '<Saved Visitor %r>' % self.id