from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

app = Flask("project")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Import rute setelah app/socketio didefinisikan
from project.controllers import *
