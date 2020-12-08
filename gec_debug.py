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
        text = "Hi, Guibin! My namme is Citao. The marked was closed yestreday. (This email are sent from OnMail.)"
    data = {
        'text': text,

        # Parameters in GECToR model
        'iterations': 3,
        'min_probability': 0.5,
        'min_error_probability': 0.7,

        # If sensitive to lower/upper case
        'case_sensitive': True,

        # If need post processing of LanguageTool
        'languagetool_post_process': True,
        # When GECToR model thinks the error probability of corrected text > <threshold>, 
        # we will call LanguageTool for post-processing.
        # Only meaningful when 'languagetool_post_process'=True
        'languagetool_call_thres': 0.7,

        # Skip corrections that contains word in whitelist
        'whitelist': ['citao', 'guibin', 'onmail'],

        # With the information every step for debugging
        'with_debug_info': True,
    }
    result = call_gec(data)
    if data['with_debug_info']:
        print(result['debug_info'])
    print('Input  :', result['input'])
    print('Output :', result['output'])
    print('Corrections :', result['corrections'])
