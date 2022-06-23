from distutils.log import debug
from unicodedata import name
import flask
from detect_mask import *
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
