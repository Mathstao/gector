import json
import spacy
import logging
import tornado.web
from time import time
from gector.gec_model import GecBERTModel
from utils.helpers import add_sents_idx
from copy import deepcopy

logging.basicConfig(format='%(levelname)s: [%(asctime)s][%(filename)s:%(lineno)d] %(message)s',level=logging.INFO)

nlp = spacy.load("en")

model = GecBERTModel(
    vocab_path = "./data/output_vocabulary",
    model_paths = ["./pretrain/roberta_1_gector.th"],
    # model_paths = ["./pretrain/bert_0_gector.th", "./pretrain/roberta_1_gector.th", "./pretrain/xlnet_0_gector.th"],
    model_name = "roberta",
    is_ensemble = False,
    iterations = 3,
)


class GECToR(tornado.web.RequestHandler):
    def post(self):
        # resp = {'status': True, 'data': '', 'corrections': [], 'orig_tokens': [], 'cor_tokens': []}
        resp ={}
        try:
            data = json.loads(self.request.body)
            text = data['text']
            config = {}
            for field in ['add_spell_check', 'iterations', 'min_probability', 'min_error_probability']:
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
            preds, probabilities_batch, idxs_batch, error_probs_batch, total_updates = model.handle_batch(full_batch=batch, config=config, debug=False)

            print(preds)
            print(probabilities_batch)
            print(idxs_batch)
            print(error_probs_batch)
            print(total_updates)

            # # singel call
            # preds = []
            # idxs_batch = []
            # for single in batch:
            #     single_preds, _, single_idxs_batch, _, cnt = model.handle_batch([single])
            #     preds.append(single_preds[0])
            #     idxs_batch.append(single_idxs_batch[0])

            # fetch correction detail
            correct_details = []
            corrections = []
            source_sents_with_idx = add_sents_idx(text, sentences)
            
            for sent, source_tokens, pred_tokens, iter_label_idxs in zip(sentences, batch, preds, idxs_batch):
                try:
                    detail = model.generate_correct_detail(sent, source_tokens, pred_tokens, iter_label_idxs)
                    correct_details.append(detail)
                except Exception as e:
                    logging.error(e, exc_info=True)
                    print('********'*5)
                    print(sent)
                    print(source_tokens)
                    print(pred_tokens)
                    print(iter_label_idxs)
            

            global_correct_details = deepcopy(correct_details)
            for sent_id, detail in enumerate(global_correct_details):
                for change in detail:
                    change[2][0] += source_sents_with_idx[sent_id][1]
                    change[2][1] += source_sents_with_idx[sent_id][1]

            for details in global_correct_details:
                for c in details:
                    corrections.append(c)

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

            resp['status'] = True
            resp['input'] = text
            resp['output'] = correct_text
            resp['corrections'] = corrections
            # resp['data'] = correct_text
            # resp['orig_tokens'] = batch
            # resp['cor_tokens'] = preds
        except:
            logging.error('Processing failed.\n******\n\n{}\n\n******'.format(text), exc_info=True)
            resp['status'] = False
            resp['input'] = text
            resp['output'] = text
            resp['corrections'] = []
        finally:
            self.write(json.dumps(resp))
            logging.info("\nInput[{}]\nOutput[{}]\nCorrections[{}]".format(text, resp['data'], resp['corrections']))

def make_app():
    return tornado.web.Application([
        (r"/correct", GECToR),
    ])


if __name__ == "__main__" :
    app = make_app()
    logging.info("start the gec server")
    app.listen(8890)
    tornado.ioloop.IOLoop.current().start()

