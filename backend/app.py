from crypt import methods
from distutils.log import debug
from sre_constants import SUCCESS
from unicodedata import name
import flask
from pkg_resources import yield_lines
from detect_mask import *
from flask import Flask, render_template , Response , request

app = Flask(__name__)



@app.route('/',methods=['GET','POST'])
def index():
    output = request.get_json()
    print(output)
    print(type(output))
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
