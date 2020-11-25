import requests
import json

import spacy
nlp = spacy.load("en")

host = "http://0.0.0.0:8890/correct"

def call_gec(data):
    resp = requests.post(host, json=data)
    res = resp.json()
    return res


if __name__ == '__main__':
    corpus = json.load(open('./bbc_news.json'))
    domains = ['business', 'entertainment', 'politics', 'sport', 'tech']
    # record = {d: {'pass': [], 'wrong':[]} for d in domains}
    for min_probability in [0.5, 0.6, 0.7, 0.8]:
        for min_error_probability in [0.7, 0.8, 0.9]:
            for add_spell_check in [True, False]:
                record = {d: [] for d in domains}
                for domain in domains:
                    print(min_probability, min_error_probability, add_spell_check, domain)
                    for ori_sent in corpus[domain][:5000]:
                        data = {
                            "text": ori_sent,
                            'iterations': 2,
                            'min_probability': min_probability,
                            'min_error_probability': min_error_probability,
                            'add_spell_check': add_spell_check,
                            'case_sensitive': True,
                        }
                        tmp_res = call_gec(data)
                        cor_sent = tmp_res['output']
                        correction = tmp_res['corrections']

                        # if ori_sent == cor_sent:
                        #     record[domain]['pass'].append(ori_sent)
                        # else:
                        #     record[domain]['wrong'].append(ori_sent)
                        record[domain].append([ori_sent, cor_sent, correction])
                json.dump(record, open('./bbc_news_result_2_{}_{}_{}.json'.format(min_probability, min_error_probability, str(add_spell_check)), 'w'))