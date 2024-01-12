import os
basedir = os.path.abspath(os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
class Config:
    # SECURITY WARNING: keep the secret key used in production secret!
    SECRET_KEY = 'django-insecure-_d3!4ap$g#wl$*t-u=43xlhv5)9y#jd*kraf-v(^0hqe_@75cg'
    EMAIL_BACKEND =os.environ.get('django.core.mail.backends.console.EmailBackend')
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    # MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    # MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    SSL_REDIRECT = False
    FROM_EMAIL_ADDRESS=os.environ.get('FROM_EMAIL_ADDRESS')
    FROM_EMAIL_ADDRESS_PASSWORD=os.environ.get('FROM_EMAIL_ADDRESS_PASSWORD')

    @staticmethod
    def init_app(app):
        pass

