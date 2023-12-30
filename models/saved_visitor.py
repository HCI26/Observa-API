from api import db

class SavedVisitor(db.Model):
    __tablename__ = 'saved_visitors'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(64))
    relationship = db.Column(db.String(64))
    image_path = db.Column(db.String)  # Store the image path
    embedding = db.Column(db.ARRAY(db.Float), nullable=False)
    last_visited = db.Column(db.DateTime)
    user = db.relationship('User', back_populates='saved_visitors', lazy=True)

    @staticmethod
    def get_saved_visitors(user_id):
        return SavedVisitor.query.filter(SavedVisitor.user_id == user_id, SavedVisitor.name.isnot(None)).all()

    @staticmethod
    def get_unknown_visitors(user_id):
        return SavedVisitor.query.filter(SavedVisitor.user_id == user_id, SavedVisitor.name.is_(None)).all()
    
    def __repr__(self):
        return '<Saved Visitor %r>' % self.id