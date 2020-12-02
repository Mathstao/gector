import json

from torch import alpha_dropout
import spacy
import logging
import requests
import tornado.web
from time import time
from gector.gec_model import GecBERTModel
from utils.helpers import add_sents_idx, add_tokens_idx
from copy import deepcopy
import errant

logging.basicConfig(format='%(levelname)s: [%(asctime)s][%(filename)s:%(lineno)d] %(message)s',level=logging.INFO)

nlp = spacy.load("en")
annotator = errant.load(lang='en', nlp=nlp)

model = GecBERTModel(
    vocab_path = "./data/output_vocabulary",
    model_paths = ["./pretrain/roberta_1_gector.th"],
    # model_paths = ["./pretrain/bert_0_gector.th", "./pretrain/roberta_1_gector.th", "./pretrain/xlnet_0_gector.th"],
    model_name = "roberta",
    is_ensemble = False,
    iterations = 3,
)

DEFAULT_CONFIG = {
    'iterations': 3,
    'min_probability': 0.5,
    'min_error_probability': 0.7,
    'case_sensitive': True,
}

LANGUAGE_TOOL_URL = 'http://localhost:8081/v2/check'

def language_tool_correct(text):
    data = {
        'language': 'en-US',
        'text': text,
    }
    resp = requests.post(LANGUAGE_TOOL_URL, data)
    result = resp.json()

    corrections = []
    for r in result["matches"]:
        idx_s = r["offset"]
        idx_e = idx_s + r["length"]
        corrections.append([text[idx_s:idx_e], r["replacements"][0]['value'], [idx_s, idx_e]])

    # generate correct text from the correction details
    offset = 0
    char_list = list(text)
    for c in corrections:
        idx_s = c[2][0] + offset
        idx_e = c[2][1] + offset
        source = list(c[0])
        target = list(c[1])
        if  char_list[idx_s:idx_e] == source:
            char_list[idx_s:idx_e] = target
            offset += len(target) - len(source)
    correct_text = ''.join(char_list)
    return correct_text, corrections


def extract_corrections_from_parallel_text(orig_text, cor_text):
    orig = annotator.parse(orig_text, True)
    cor = annotator.parse(cor_text, True)
    corrections = extract_corrections_from_parallel_tokens(orig_text, orig, cor)
    return corrections


def extract_corrections_from_parallel_tokens(orig_text, orig, cor):
    edits = annotator.annotate(orig, cor, need_tag=False)
    orig_tokens_with_idx = add_tokens_idx(orig_text, orig)
    orig_tokens_with_idx
    corrections = []
    for e in edits:
        o_char_start = orig_tokens_with_idx[e.o_start][1]
        o_char_end = orig_tokens_with_idx[e.o_end-1][2]
        orig_substr = e.o_str
        pred_substr = e.c_str
        corrections.append([orig_substr, pred_substr, [o_char_start, o_char_end]])
    return corrections


def apply_corrections(text, corrections):
    # generate correct text from the correction details
    offset = 0
    char_list = list(text)
    for c in corrections:
        idx_s = c[2][0] + offset
        idx_e = c[2][1] + offset
        source = list(c[0])
        target = list(c[1])
        if  char_list[idx_s:idx_e] == source:
            char_list[idx_s:idx_e] = target
            offset += len(target) - len(source)
    correct_text = ''.join(char_list)
    return correct_text


class GECToR(tornado.web.RequestHandler):
    def post(self):
        resp ={}
        try:
            data = json.loads(self.request.body)
            text = data['text']
            config = DEFAULT_CONFIG.copy()
            for field in ['iterations', 'min_probability', 'min_error_probability', 'case_sensitive']:
                if field in data:
                    config[field] = data[field]

            # tokenize the input text
            batch = []
            doc = nlp(text)
            sentences = []
            for sent in doc.sents:
                tokens = []
                sentences.append(sent.text)
                for token in nlp(sent.text):
                    tokens.append(token.text)
                tokens = [i.strip() for i in tokens if i.strip()]
                batch.append(tokens)

            # batch call
            result = model.handle_batch(full_batch=batch, config=config)
            pred_tokens_batch = result['pred_tokens_batch']
            edit_probas_batch = result['edit_probas_batch']
            edit_idxs_batch = result['edit_idxs_batch']
            inter_pred_tokens_batch = result['inter_pred_tokens_batch']
            error_probs_batch = result['error_probs_batch']
            last_error_prob_batch = result['last_error_prob_batch']

            # origanize the debuging information.
            debug_text_output_list = []
            for ori_token, curr_error_probs, curr_edit_idxs, curr_edit_probs, curr_iter_pred in zip(batch, error_probs_batch, edit_idxs_batch, edit_probas_batch, inter_pred_tokens_batch):
                inter_corrected_tokens = [ori_token] + curr_iter_pred
                inter_corrected_tokens = [['__START__']+i for i in inter_corrected_tokens]
                for i, (iter_error_prob, iter_idxs, iter_probs, iter_pred) in enumerate(zip(curr_error_probs, curr_edit_idxs, curr_edit_probs, curr_iter_pred)):
                    iter_labels = [model.vocab.get_token_from_index(i, namespace='labels') for i in iter_idxs]
                    debug_text_output_list.append('<Iteration {}>'.format(i+1))
                    debug_text_output_list.append('Sentence Error Probability: {}\n'.format(iter_error_prob))
                    debug_text_output_list.append('#### Before ####\n{}\n'.format(' '.join(inter_corrected_tokens[i][1:])))
                    debug_text_output_list.append('#### After #####\n{}\n'.format(' '.join(inter_corrected_tokens[i+1][1:])))
                    debug_text_output_list.append('[Model Correction]')
                    debug_text_output_list.append('-'*80)
                    for _token, _edit, _proba in zip(inter_corrected_tokens[i], iter_labels, iter_probs):
                        debug_text_output_list.append(_token.ljust(30)+_edit.ljust(30)+str(_proba))
                    debug_text_output_list.append('-'*80)
                    debug_text_output_list.append('\n')
                debug_text_output_list.append('='*80+'\n')
            debug_text_output = '\n'.join(debug_text_output_list)
            del debug_text_output_list

            # fetch correction detail
            local_gec_corrections_list = []
            source_sents_with_idx = add_sents_idx(text, sentences)
            for sent, source_tokens, pred_tokens, iter_label_idxs, iter_probs, iter_error_probs in zip(sentences, batch, pred_tokens_batch, edit_idxs_batch, edit_probas_batch, error_probs_batch):
                # detail = model.generate_correct_detail(sent, source_tokens, pred_tokens, iter_label_idxs, iter_probs, iter_error_probs, config)
                gec_corrections = extract_corrections_from_parallel_tokens(sent, source_tokens, pred_tokens)
                gec_cor_sent = apply_corrections(sent, gec_corrections)
                local_gec_corrections_list.append(gec_corrections)

            global_gec_corrections_list = deepcopy(local_gec_corrections_list)
            for sent_id, corrections in enumerate(global_gec_corrections_list):
                for change in corrections:
                    change[2][0] += source_sents_with_idx[sent_id][1]
                    change[2][1] += source_sents_with_idx[sent_id][1]

            global_gec_corrections = []
            for corrections in global_gec_corrections:
                global_gec_corrections.extend(corrections)

            # generate correct text from the correction details
            correct_text = apply_corrections(text, global_gec_corrections)

            # Add LanguageTool result
            resp['status'] = True
            resp['input'] = text
            resp['output'] = correct_text
            resp['corrections'] = global_gec_corrections
            resp['lt_correct_text'], _ = language_tool_correct(correct_text)
            resp['lt_corrections'] = extract_corrections_from_parallel_text(text, resp['lt_correct_text'])
            if config.get('debug_info', True):
                resp['debug_info'] = debug_text_output
            else:
                resp['debug_info'] = ''
            white_list = ['Citao', 'OnMail']
            filtered_corrections = []
            for c in resp['lt_corrections']:
                skip_flag = False
                for word in white_list:
                    word = word.lower()
                    if word in c[0].lower():
                        skip_flag = True
                        break
                if not skip_flag:
                    filtered_corrections.append(c)
            resp['filtered_corrections'] = filtered_corrections
            resp['final_output'] = apply_corrections(text, resp['filtered_corrections'])
            
                        
                        
            
        except:
            logging.error('Processing failed.\n******\n\n{}\n\n******'.format(text), exc_info=True)
            resp['status'] = False
            resp['input'] = text
            resp['output'] = text
            resp['corrections'] = []
            resp['debug_info'] = ''
            resp['lt_correct_text'] = text
            resp['lt_corrections'] = []
            resp['filtered_corrections'] = []
            resp['final_output'] = text
        finally:
            self.write(json.dumps(resp))
            logging.info("\nInput  [{}]\nOutput [{}]\nCorrections [{}]".format(resp['input'], resp['output'], resp['corrections']))

def make_app():
    return tornado.web.Application([
        (r"/correct", GECToR),
    ])


if __name__ == "__main__" :
    app = make_app()
    logging.info("start the gec server")
    app.listen(8890)
    tornado.ioloop.IOLoop.current().start()

