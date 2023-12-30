from api import mail
from flask_mail import Message

def send_notification(subject, recepient_email, body):
    msg = Message(subject=subject, sender='observah60@gmail.com', recipients=[recepient_email])
    msg.body = body
    mail.send(msg)
    return True