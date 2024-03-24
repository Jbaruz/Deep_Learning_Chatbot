from flask import Flask

app = Flask(__name__)
print("Initializing app")

from app import routes

