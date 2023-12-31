from random import randint
from sqlalchemy.exc import IntegrityError
from faker import Faker
from . import db
from models import User, SavedVisitor


def users(count=100):
    fake = Faker()
    i = 0
    while i < count:
        u = User(username=fake.user_name(),
                 passhash='password',
                 email=fake.email(),
                 phone_number=fake.msisdn(),
                 full_name=fake.name_male(),
                 date_of_birth=fake.date_of_birth(minimum_age=18),
                 city=fake.city(),
                 country=fake.country(),
                 gender="male",
                 address=fake.street_address())
                 
        db.session.add(u)
        try:
            db.session.commit()
            i += 1
        except IntegrityError:
            db.session.rollback()