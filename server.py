import json

from torch import alpha_dropout
import spacy
import logging
import requests
import tornado.web
from time import time
from gector.gec_model import GecBERTModel
from utils.helpers import add_sents_idx, add_tokens_idx, token_level_edits, forward_merge_corrections, backward_merge_corrections
from copy import deepcopy
import pprint
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
    'languagetool_post_process': True,
    'languagetool_call_thres': 0.7,
    'whitelist': [],
    'with_debug_info': True
}

LANGUAGE_TOOL_URL = 'http://localhost:8081/v2/check'
UNKNOWN_LABEL_INDEX = 5000


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
        # Limit the number of 'replacements'
        r["replacements"] = r["replacements"][:5]
    correct_text = apply_corrections(text, corrections)
    return correct_text, corrections, result


def extract_corrections_from_parallel_text(orig_text, cor_text):
    # orig = annotator.parse(orig_text, True)
    # cor = annotator.parse(cor_text, True)
    orig = [i.text for i in nlp(orig_text)]
    cor = [i.text for i in nlp(cor_text)]
    corrections = extract_corrections_from_parallel_tokens(orig_text, orig, cor)
    return corrections


# def extract_corrections_from_parallel_tokens(orig_text, orig, cor):
#     # print(orig_text)
#     # print(orig)
#     # print(cor)
#     if isinstance(orig, list):
#         orig = annotator.parse(' '.join(orig), True)
#     if isinstance(cor, list):
#         cor = annotator.parse(' '.join(cor), True)
#     edits = annotator.annotate(orig, cor, merging="rules", need_tag=False)
#     orig_tokens_with_idx = add_tokens_idx(orig_text, orig)
#     # print(orig_tokens_with_idx)
#     orig_tokens_with_idx
#     corrections = []
#     for e in edits:
#         # print(e.__str__())
#         o_char_start = orig_tokens_with_idx[e.o_start][1]
#         o_char_end = orig_tokens_with_idx[e.o_end-1][2]
#         orig_substr = e.o_str
#         pred_substr = e.c_str
#         corrections.append([orig_substr, pred_substr, [o_char_start, o_char_end]])
#     return corrections


def extract_corrections_from_parallel_tokens(orig_text, orig, cor):
    """
    corrections:
    [['I', 'I', [0, 1]],
    ['went', 'went to', [2, 6]],
    ['school', 'school', [7, 13]],
    ['this ysterday', 'yesterday ', [14, 27]]]
    """
    _, _, _, edits = token_level_edits(orig, cor)
    align_orig_tokens = [i[0] for i in edits]
    align_cor_tokens = [i[1] for i in edits]
    
    orig_tokens_with_index = add_tokens_idx(orig_text, orig)
    align_orig_tokens_with_index = add_tokens_idx(orig_text, align_orig_tokens)

    corrections = []
    for orig_token, pred_token in zip(align_orig_tokens_with_index, align_cor_tokens):
        corrections.append([orig_token[0], pred_token, [orig_token[1][0], orig_token[1][1]]])
    corrections = backward_merge_corrections(orig_text, corrections)
    corrections = forward_merge_corrections(orig_text, corrections)
    
    # fix minor space issue
    for i, (o, c) in enumerate(zip(orig_tokens_with_index, corrections)):
        corrections[i] = [c[0].strip(), c[1], o[1]]

    # filtering
    corrections = [c for c in corrections if c[0]!=c[1]]
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


def filter_white_corrections(corrections, whitelist):
    if not whitelist:
        return corrections
    filtered_corrections = []
    for c in corrections:
        skip_flag = False
        for word in whitelist:
            word = word.lower()
            if word in c[0].lower():
                skip_flag = True
                break
        if not skip_flag:
            filtered_corrections.append(c)
    return filtered_corrections


def merge_adjacent_corrections(corrections):
    cor_id = 0
    last_end_idx = 0
    accumulated = []
    merged_corrections = []
    while cor_id < len(corrections):
        curr_cor = corrections[cor_id]
        if last_end_idx != 0 and curr_cor[2][0] == last_end_idx:
            accumulated.append(curr_cor)
            last_end_idx = curr_cor[2][1]
        else:
            if accumulated:
                merged_corrections.append([
                    ''.join(x[0] for x in accumulated),
                    ''.join(x[1] for x in accumulated),
                    [accumulated[0][2][0], accumulated[-1][2][1]]
                ])
            accumulated = [curr_cor]
            last_end_idx = curr_cor[2][1]
        cor_id += 1
    if accumulated:
        merged_corrections.append([
            ''.join(x[0] for x in accumulated),
            ''.join(x[1] for x in accumulated),
            [accumulated[0][2][0], accumulated[-1][2][1]]
        ])
    return merged_corrections


class GECToR(tornado.web.RequestHandler):
    def post(self):
        resp ={}
        data = json.loads(self.request.body)
        text = data['text']
        config = DEFAULT_CONFIG.copy()

        for field in ['iterations', 'min_probability', 'min_error_probability', 'case_sensitive', 'languagetool_post_process', 'languagetool_call_thres', 'whitelist', 'with_debug_info']:
            if field in data:
                config[field] = data[field]
        try:
            # Tokenize the input text
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

            # Batch call
            result = model.handle_batch(full_batch=batch, config=config)
            pred_tokens_batch = result['pred_tokens_batch']
            edit_probas_batch = result['edit_probas_batch']
            edit_idxs_batch = result['edit_idxs_batch']
            inter_pred_tokens_batch = result['inter_pred_tokens_batch']
            error_probs_batch = result['error_probs_batch']
            last_error_prob_batch = result['last_error_prob_batch']

            debug_text_output_list = []

            # Fetch correction detail
            local_corrections_list = []
            source_sents_with_idx = add_sents_idx(text, sentences)
            for sent_id, (sent, source_tokens, pred_tokens, iter_label_idxs, iter_probs, curr_iter_pred, iter_error_probs, last_error_prob) in enumerate(zip(
                sentences, batch, pred_tokens_batch, edit_idxs_batch, edit_probas_batch, inter_pred_tokens_batch, error_probs_batch, last_error_prob_batch)):

                # Skip
                if iter_label_idxs == []:
                    local_corrections_list.append([])
                    continue

                gec_corrections = extract_corrections_from_parallel_tokens(sent, source_tokens, pred_tokens)
                print('gec_corrections', gec_corrections)
                gec_correct_text = apply_corrections(sent, gec_corrections)
                print('gec_correct_text', gec_correct_text)

                # Origanize the debuging information.
                if config['with_debug_info']:
                    debug_text_output_list.append('{} Sentence_{} {}'.format('='*34, sent_id+1, '='*34))
                    debug_text_output_list.append('[GECToR Model]')
                    inter_corrected_tokens = [source_tokens] + curr_iter_pred
                    inter_corrected_tokens = [['__START__']+i for i in inter_corrected_tokens]
                    for i, (iter_error_prob, iter_idxs, iter_probs) in enumerate(zip(iter_error_probs, iter_label_idxs, iter_probs)):
                        iter_labels = [model.vocab.get_token_from_index(i, namespace='labels') for i in iter_idxs]
                        debug_text_output_list.append('<Iteration {}>'.format(i+1))
                        debug_text_output_list.append('Sentence Error Probability: {}'.format(iter_error_prob))
                        debug_text_output_list.append('# From #  {}'.format(' '.join(inter_corrected_tokens[i][1:])))
                        debug_text_output_list.append('#  To  #  {}'.format(' '.join(inter_corrected_tokens[i+1][1:])))
                        debug_text_output_list.append('-'*80)
                        for _token, _edit, _proba in zip(inter_corrected_tokens[i], iter_labels, iter_probs):
                            debug_text_output_list.append(_token.ljust(30)+_edit.ljust(30)+str(_proba))
                        debug_text_output_list.append('-'*80+'\n')
                # Condiftions to call LanguageTool
                if config['languagetool_post_process'] and \
                    (last_error_prob > config['languagetool_call_thres'] or \
                     (len(iter_label_idxs)>0 and UNKNOWN_LABEL_INDEX in iter_label_idxs[-1])):
                    logging.info('sentence[{}] need_lt[{}] edits[{}] error_prob[{}] need call LanguageTool'.format(gec_correct_text, config['languagetool_post_process'], iter_label_idxs[-1], last_error_prob))
                    # lt_correct_text, _ = language_tool_correct(gec_correct_text)
                    # lt_corrections = extract_corrections_from_parallel_text(sent, lt_correct_text)
                    lt_correct_text, lt_corrections, lt_result = language_tool_correct(sent)
                    if config['with_debug_info']:
                        debug_text_output_list.append('[LanguageTool]')
                        debug_text_output_list.append('# From #  {}'.format(gec_correct_text))
                        debug_text_output_list.append('#  To  #  {}'.format(lt_correct_text))
                        debug_text_output_list.append('-'*80)
                        debug_text_output_list.append(pprint.pformat(lt_result['matches'], indent=2))
                        debug_text_output_list.append('-'*80+'\n')
                else:
                    logging.info('sentence[{}] need_lt[{}] edits[{}] error_prob[{}] skip calling LanguageTool'.format(gec_correct_text, config['languagetool_post_process'], iter_label_idxs[-1], last_error_prob))
                    lt_correct_text = gec_correct_text
                    lt_corrections = gec_corrections
                
                final_corrections = extract_corrections_from_parallel_text(sent, lt_correct_text)

                # Filter corrections related to whitelist
                final_corrections = filter_white_corrections(final_corrections, config['whitelist'])
                # Merge adjacent corrections
                final_corrections = merge_adjacent_corrections(final_corrections)
                
                local_corrections_list.append(final_corrections)

            # Debugging layout
            debug_text_output = '\n'.join(debug_text_output_list)
            del debug_text_output_list

            # Convert local corrections into global corrections
            global_corrections_list = deepcopy(local_corrections_list)
            for sent_id, corrections in enumerate(global_corrections_list):
                for change in corrections:
                    change[2][0] += source_sents_with_idx[sent_id][1][0]
                    change[2][1] += source_sents_with_idx[sent_id][1][0]

            global_corrections = []
            for corrections in global_corrections_list:
                global_corrections.extend(corrections)

            # Generate correct text from the correction details
            correct_text = apply_corrections(text, global_corrections)

            resp['status'] = True
            resp['input'] = text
            resp['output'] = correct_text
            resp['corrections'] = global_corrections
            resp['debug_info'] = debug_text_output
            
        except:
            logging.error('Processing failed.\n******\n\n{}\n\n******'.format(text), exc_info=True)
            resp['status'] = False
            resp['input'] = text
            resp['output'] = text
            resp['corrections'] = []
            resp['debug_info'] = ''
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

