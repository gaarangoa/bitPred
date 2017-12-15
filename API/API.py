from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse, abort, Resource
import predict
import json
local_dir = "/var/www/html/bitpred/API";
p = predict.Predict(model_name = local_dir+'/model.hdf5')

# parser = reqparse.RequestParser()
# parser.add_argument("task")

app = Flask(__name__)
api = Api(app)

class PredictSentiment(Resource):
    def get(self):
        return {'info': 'This API is designed to predict the sentiment (bearish/bullish) of bitcoin data'}
    def post(self):
        args = request.get_json(force=True)
        # print(args)
        return jsonify(p.pred(query = args))

api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True, port=5055)

