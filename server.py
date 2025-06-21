from project.controllers.hello import app, socketio

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
