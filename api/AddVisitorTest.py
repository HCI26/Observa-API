
#note: this is a test file for the add visitor function. It is not a part of the main code.
# pip install Flask pytest mock




import pytest
import os
from unittest.mock import patch
from your_flask_app import create_app, db  # Adjust this import according to your app structure
from your_flask_app.models import SavedVisitor  # Adjust this import according to your app structure

# Setup for Flask application context
@pytest.fixture
def client():
    app = create_app({'TESTING': True, 'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_add_visitor_with_valid_input(client):
    with patch('os.path.isfile', return_value=True), \
         patch('DeepFace.represent', return_value=[{"embedding": [0.1, 0.2, ...]}]), \
         patch('your_flask_app.db.session.add') as mock_db_add, \
         patch('your_flask_app.db.session.commit') as mock_db_commit:
        
        response = client.post('/path_to_add_visitor', json={
            'name': 'John Doe',
            'relationship': 'friend',
            'user_id': '123',
            'img_url': '/path/to/image.jpg'
        })
        
        assert response.status_code == 201
        assert response.json['message'] == "new visitor added successfully"
        mock_db_add.assert_called()
        mock_db_commit.assert_called()

def test_add_visitor_with_missing_input(client):
    response = client.post('/path_to_add_visitor', json={
        'name': 'John Doe',
        'user_id': '123'
        # Missing relationship and img_url
    })
    
    assert response.status_code == 400
    assert response.json['message'] == "you must pass name, relationship, user_id, and img_url inputs"

def test_add_visitor_with_invalid_img_path(client):
    with patch('os.path.isfile', return_value=False):
        response = client.post('/path_to_add_visitor', json={
            'name': 'John Doe',
            'relationship': 'friend',
            'user_id': '123',
            'img_url': '/invalid/path.jpg'
        })

        assert response.status_code == 400
        assert response.json == "img_path does not exist"

def test_add_visitor_no_face_detected(client):
    with patch('os.path.isfile', return_value=True), \
         patch('DeepFace.represent', return_value=[]):  # Simulating no face detected
        
        response = client.post('/path_to_add_visitor', json={
            'name': 'John Doe',
            'relationship': 'friend',
            'user_id': '123',
            'img_url': '/path/to/image.jpg'
        })

        assert response.status_code == 400
        assert response.json == "no face detected in image"
