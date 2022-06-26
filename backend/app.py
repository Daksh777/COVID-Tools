# from crypt import methods
# from distutils.log import debug
# from sre_constants import SUCCESS
# from unicodedata import name
# import flask
# from pkg_resources import yield_lines
from heart_rate import *
from detect_mask import *
from flask import Flask, render_template , Response , request , jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mask',methods=['GET','POST'])
def index1():
    return render_template('mask.html')

@app.route('/heart_rate',methods=['GET','POST'])
def index2():
    return render_template('heart.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
