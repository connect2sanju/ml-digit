from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/flask')
def hello():
    return 'Hello, World from flask!'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return 'testing'
    else:
        return 'not required!'