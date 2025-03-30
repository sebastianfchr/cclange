# import argparse
# import requests

# parser = argparse.ArgumentParser("benchmarkparser")
# parser.add_argument("--reqserver", help="A server whose <<server>>/predict_sentence/ and <<server>>/predict_sentence_batch/ will receive POST Requests for benchmarking", type=str)
# args = parser.parse_args()
# print(args.reqserver)
# # print(args.counter + 1)


# # URL of the FastAPI server
# # url = "http://0.0.0.0:8001/predict_sentence_batch/".format(args.reqserver)
# url = "https://0.0.0.0:8001/"

# # Send the POST request with the data as JSON
# # response = requests.post(url, json=data)
# response = requests.get(url)

# # # Handle the response
# # if response.status_code == 200:
# #     print("Request successful!")
# #     print("Response JSON:", response.json())
# # else:
# #     print(f"Request failed with status code {response.status_code}")


import requests
import json

# url = "http://localhost:8001/predict_sentence_batch/"
url = "http://localhost:8001/"
data = {
    "sentences": [
        "Going down the beautiful road, I met a horrible rabbit",
        "while drinking a craft beer, I became damn hungry"
    ],
    "sentiments": ["negative", "neutral"]
}

headers = {
    'Content-Type': 'application/json'
}

try:
    response = requests.get(url, json=data, headers=headers)
    response.raise_for_status()  # Raise an exception for bad HTTP status codes
    print("Response:", response.json())  # Check response body
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.RequestException as err:
    print(f"Error occurred: {err}")