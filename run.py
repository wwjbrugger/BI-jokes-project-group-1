#! /usr/bin/env python

import biapp
from biapp import app
if __name__ == "__main__":
    app.secret_key = "ABCDEFF"
    app.run(host='0.0.0.0')
    #app.run(debug=True)

"""
if __name__ == "__main__":
    app.secret_key = "ABCDEFF"
    app.run(host='192.168.0.6', debug=True)
"""