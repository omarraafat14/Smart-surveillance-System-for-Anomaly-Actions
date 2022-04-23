import json as j
import requests
from flask import Flask, request
from flask_restful import Resource, Api
import base64
from PIL import Image
from io import BytesIO
import prediction
app = Flask(__name__)
api = Api(app)

class Result(Resource):
	def get(self):

		res = prediction.predict_single_action("https://youtu.be/jEexefuB62c")
		with open(R"D:\DataSet\exp\Explosion032_x264.mp4_frame70.jpg", "rb") as image_file:
			imge = base64.b64encode(image_file.read())

		classes = list(res.keys())[0],
		data1 = {"value": imge ,"dt": "03/04/022","zone": "0","anomalyType": classes}
		r = requests.post('https://datapostapi.conveyor.cloud/api/Values/',data = data1)
		#print(r.json())
		props=list(res.values()),
		return "DONE!!!!!!!"

api.add_resource(Result, '/')
if __name__ == '__main__':
	app.run(debug=True)