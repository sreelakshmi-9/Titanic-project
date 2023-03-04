# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 22:01:43 2022

@author: Sree
"""


import json
import requests

url = 'http://127.0.0.1:5000'
headers = {'Content-type':'application/json'}


request_data = json.dumps({"data":[[7.4,0.8,0, 1.9, 0.07, 10, 34, 0.9978, 3.51, 0.56, 9.4]]})
response = requests.post(url, request_data, headers=headers)
print("Predicted Label")
print(response.text)  
