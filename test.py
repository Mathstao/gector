import os
import requests
from pprint import pprint

hosts = {
    1: "http://0.0.0.0:8891/correct",
    2: "http://0.0.0.0:8892/correct",
    3: "http://0.0.0.0:8893/correct",
    4: "http://0.0.0.0:8894/correct",
    5: "http://0.0.0.0:8895/correct",
}

def call_gec(host, text, add_spell_check=False):
    data = { "text": text , 'add_spell_check': add_spell_check }
    resp = requests.post(host, json=data)
    res = resp.json()
    return res

ref_m2_files = [
    '/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/wi+locness/m2/ABCN.all.gold.bea19.m2'
]

ref_m2 = []
for rmf in ref_m2_files:
    ref_m2.extend(open(rmf).read().split('\n\n'))

orig_sents = [i.split('\n')[0].lstrip('S ') for i in ref_m2]

orig_file = './output/original.txt'
of = open(orig_file, 'w')
for sent in orig_sents:
    of.write(sent+'\n')
of.close()


for flag in ['', '_spell_check']:
    if flag == '_spell_check':
        add_spell_check = True
    else:
        add_spell_check = False

    for _iter in [1, 2, 5]:
        host = hosts[_iter]
        cor_file = './output/wil_corrected_iter{}{}.txt'.format(str(_iter), flag)
        cor_m2_file = './output/wil_corrected_iter{}{}.m2'.format(str(_iter), flag)
        cf = open(cor_file, 'w')
        for sent in orig_sents:
            res = call_gec(host, sent, add_spell_check)
            cf.write(res['data']+'\n')
        cf.close()

        py_env_bin = '/home/citao/anaconda2/envs/python37/bin/'
        errant_parallel_bin = os.path.join(py_env_bin, 'errant_parallel')
        shell_cmd = "{} -orig {} -cor {} -out {}".format(errant_parallel_bin, orig_file, cor_file, cor_m2_file)
        print(shell_cmd)
        print(os.popen(shell_cmd).read())


