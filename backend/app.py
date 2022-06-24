from crypt import methods
from distutils.log import debug
from sre_constants import SUCCESS
from unicodedata import name
import flask
from pkg_resources import yield_lines
from detect_mask import *
from flask import Flask, render_template , Response , request , jsonify

app = Flask(__name__)



@app.route('/',methods=['GET','POST'])
def index():
    if request.headers.get('Content-Type') == 'video/mp4':
        # load the full request data into memory
        # rawdata = request.get_data()
        
        # or use the stream 
        rawdata = request.stream.read()

        # process the data here!

        return jsonify(success=True)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
