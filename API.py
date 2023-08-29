from flask import Flask, request, jsonify
import requests
import torch
from PIL import Image

app = Flask(__name__)

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')

# Import the prepair function
from prepair import prepair


@app.route('/api/solve', methods=['GET'])
def solve():
    # Get the image URL from the query parameters
    image_url = request.args.get('image_url')

    # Download and open the image using PIL
    img = Image.open(requests.get(image_url, stream=True).raw)

    # Preprocess the image using the prepair function
    img = prepair(img)

    # Perform inference with the model
    result = model(img)
    a = result.pandas().xyxy[0].sort_values('xmin')

    while len(a) > 6:
        lines = a.confidence
        linev = min(a.confidence)
        for line in lines.keys():
            if lines[line] == linev:
                a = a.drop(line)

    # Generate the result string
    result_str = ""
    for _, key in a.name.items():
        result_str += key

    # Return the result as JSON
    return jsonify(result=result_str)


if __name__ == '__main__':
    app.run()
