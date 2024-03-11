from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
import numpy as np
from tensorflow import keras
from numpy import asarray
# self lib

from api_model.LandingPage import LandingPage
from api_model.Kelas import Kelas
from api_model.Scanner import Scanner

# from Kelas import Kelas
# from Scanner import Scanner
# from LandingPage import LandingPage


app = Flask(__name__)
api = Api(app)

# Api endpoints
api.add_resource(LandingPage, "/")
api.add_resource(Scanner, "/scanner")
api.add_resource(Kelas, "/kelas")

# Start server
# Lets use debug for development

PORT = 8080
if __name__ == '__main__':
    app.run(debug = False, host='0.0.0.0', port=PORT)
