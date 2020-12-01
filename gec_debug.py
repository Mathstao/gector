import sys
import requests

host = "http://11.0.0.150:8890/correct"

def call_gec(data):
    resp = requests.post(host, json=data)
    res = resp.json()
    return res

if __name__ == '__main__':
    if len(sys.argv)==2:
        text = sys.argv[-1]
    else:
        text = "Wjere is yourr from?"
    data = {
        "text": text,
        'iterations': 3,
        'min_probability': 0.5,
        'min_error_probability': 0.8,
        'debug': True,
    }
    result = call_gec(data)
    if data['debug']:
        print(result['debug_info'])
        print('\n')
    print('Input  :', result['input'])
    print('Output :', result['output'])
    print('Corrections :', result['corrections'])
