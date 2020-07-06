import json
import requests
import pandas as pd
from oldcare.utils.pathassistant import get_path

people_info_path = get_path('people_info_path')


def get_response(url, request):
    headers = {'content-type': 'application/json'}
    ret = requests.post(url, json=request, headers=headers)
    if ret.status_code == 200:
        text = json.loads(ret.text)
        return text
    else:
        return 'error'


def get_people_info():
    url = "http://localhost:10000/else/queryAll"
    request = {}
    response = get_response(url, request)
    if response == 'error':
        print('error')
    else:
        data = response['data']
        data.append({'id_card': 'Unknown', 'name': '陌生人', 'type': 'stranger'})
        df = pd.DataFrame(data)
        df.to_csv(people_info_path, index=False)
