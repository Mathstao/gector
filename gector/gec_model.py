"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time
import numpy as np
import torch
from typing_extensions import final
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.seq2labels_model import Seq2Labels
from gector.wordpiece_indexer import PretrainedBertIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, add_tokens_idx, START_TOKEN


logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            logging.warning('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            logging.warning('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        logging.warning('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'


class GecBERTModel(object):
    def __init__(self, vocab_path=None, model_paths=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=3,
                 min_probability=0.0,
                 model_name='roberta',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 resolve_cycles=False,
                 ):
        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(model_paths)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_probability = min_probability
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles
        # set training parameters and operations

        self.indexers = []
        self.models = []
        for model_path in model_paths:
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                               confidence=self.confidence
                               ).to(self.device)
            logging.info('Loading {}'.format(model_path))
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path,
                                                 map_location=torch.device('cpu')))
            model.eval()
            self.models.append(model)

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def _restore_model(self, input_path):
        if os.path.isdir(input_path):
            logging.error("Model could not be restored from directory: {}".format(input_path))
            filenames = []
        else:
            filenames = [input_path]
        for model_path in filenames:
            try:
                if torch.cuda.is_available():
                    loaded_model = torch.load(model_path)
                else:
                    loaded_model = torch.load(model_path,
                                              map_location=lambda storage,
                                                                  loc: storage)
            except:
                logging.error("{} is not valid model".format(input_path))
            own_state = self.model.state_dict()
            for name, weights in loaded_model.items():
                if name not in own_state:
                    continue
                try:
                    if len(filenames) == 1:
                        own_state[name].copy_(weights)
                    else:
                        own_state[name] += weights
                except RuntimeError:
                    continue
        logging.info("Model is restored")

    def predict(self, batches):
        t11 = time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                prediction = model.forward(**batch)
                for k in ['class_probabilities_labels', 'class_probabilities_d_tags']:
                    v = prediction[k]
#                     # prediction detail [Citao]
#                     print(k)
#                     print(v.shape)
#                     print(v)
#                     for i in v[0]:
#                         print(np.argsort(list(i))[::-1][:20])
#                     print('-')
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)
        t55 = time()
        if self.log:
            logging.info("Inference time {}".format(t55-t11))
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token, min_probability):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < min_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def _get_embbeder(self, weigths_name, special_tokens_fix):
        embedders = {'bert': PretrainedBertEmbedder(
            pretrained_model=weigths_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=special_tokens_fix)
        }
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True)
        return text_field_embedder

    def _get_indexer(self, weights_name, special_tokens_fix):
        bert_token_indexer = PretrainedBertIndexer(
            pretrained_model=weights_name,
            do_lowercase=self.lowercase_tokens,
            max_pieces_per_token=5,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            special_tokens_fix=special_tokens_fix,
            is_test=True
        )
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]
                tokens = [Token(token) for token in ['$START'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)
            batch.index_instances(self.vocab)
            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs, min_probability, min_error_probability,
                          max_len=50):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            length = min(len(tokens), max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i], namespace='labels')
                action = self.get_token_action(token, i, probabilities[i], sugg_token, min_probability)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch, config): 
        """
        Handle batch of requests.
        """
        # overwrite the predict config
        iterations = config.get('iterations', self.iterations)
        min_probability = config.get('min_probability', self.min_probability)
        min_error_probability = config.get('min_error_probability', self.min_error_probability)

        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch)) if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        correct_cnt = 0
        last_error_prob_batch = []

        probabilities_batch = [[] for i in range(len(final_batch))]
        idxs_batch = [[] for i in range(len(final_batch))]
        error_probs_batch = [[] for i in range(len(final_batch))]
        inter_pred_batch = [[] for i in range(len(final_batch))]


        for n_iter in range(iterations):
            orig_batch = [final_batch[i] for i in pred_ids]
            sequences = self.preprocess(orig_batch)
            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences)
            pred_batch = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs,
                                                min_probability, min_error_probability)
            for pid, to_add in zip(pred_ids, probabilities):
                probabilities_batch[pid].append(to_add)
            for pid, to_add in zip(pred_ids, error_probs):
                error_probs_batch[pid].append(to_add)
            for pid, to_add in zip(pred_ids, idxs):
                idxs_batch[pid].append(to_add)
            for pid, to_add in zip(pred_ids, pred_batch[:]):
                inter_pred_batch[pid].append(to_add)

            if self.log:
                logging.info(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")


            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)

            correct_cnt += cnt

            if not pred_ids:
                break

        # Add last_error_prob_batch for post-processing
        sequences = self.preprocess(final_batch)
        _, _, last_error_prob_batch = self.predict(sequences)
        print(last_error_prob_batch)

        ### Citao added ###

        for sent_id in range(len(idxs_batch)):
            L = len(full_batch[sent_id])+1
            for iter_id, iter_idxs in enumerate(idxs_batch[sent_id]):
                idxs_batch[sent_id][iter_id] = idxs_batch[sent_id][iter_id][:L]
                edit_names = [self.vocab.get_token_from_index(i, namespace='labels') for i in iter_idxs]
                delete_num = sum([1 if i.startswith('$DELETE') else 0 for i in edit_names])
                append_num = sum([1 if i.startswith('$APPEND') else 0 for i in edit_names])
                L = L + (append_num - delete_num)

        ###################

        result = {
            'pred_tokens_batch': final_batch,
            'edit_probas_batch': probabilities_batch,
            'edit_idxs_batch': idxs_batch,
            'inter_pred_tokens_batch': inter_pred_batch,
            'error_probs_batch': error_probs_batch,
            'last_error_prob_batch': last_error_prob_batch,
            'correct_cnt': correct_cnt,

        }
        return result


    def overlay_edit(self, placeholder_list, label_idxs):
        new_placeholder_list = []
        label_names = [self.vocab.get_token_from_index(i, namespace='labels') for i in label_idxs]
        token_ptr = 0
        label_ptr = 0
        
        while label_ptr < len(label_names) and token_ptr < len(placeholder_list):

            if placeholder_list[token_ptr] == '__deleted__':
                new_placeholder_list.append('__deleted__')
                token_ptr += 1
                continue

            label = label_names[label_ptr]

            if label == '$KEEP':
                new_placeholder_list.append(placeholder_list[token_ptr])
                label_ptr += 1
                token_ptr += 1

            elif label == '$DELETE':
                meta = '__deleted__'
                if placeholder_list[token_ptr] != '__appended__':
                    new_placeholder_list.append(meta)
                label_ptr += 1
                token_ptr += 1

            elif label.startswith('$REPLACE') or label.startswith('$TRANSFORM'):
                new_placeholder_list.append('__changed__')
                label_ptr += 1
                token_ptr += 1
                
            elif label.startswith('$APPEND'):
                new_placeholder_list.extend([placeholder_list[token_ptr], '__appended__'])
                label_ptr += 1
                token_ptr += 1

            elif label in ['@@UNKNOWN@@', '@@PADDING@@']:
                logging.info('Specific label [{}]'.format(label))
                new_placeholder_list.append(placeholder_list[token_ptr])
                label_ptr += 1
                token_ptr += 1

            else:
                logging.info('Unknown label [{}]'.format(label))
                new_placeholder_list.append(placeholder_list[token_ptr])
                label_ptr += 1
                token_ptr += 1
        
        new_placeholder_list.extend(placeholder_list[token_ptr:])

        return new_placeholder_list

    def generate_edits_based_pred(self, pred_tokens, placeholder_list):
        pred_tokens = ['__START__'] + pred_tokens
        selected_placeholder_list = [i for i in placeholder_list if i != '__deleted__']
        result = []
        for token, ph in zip(pred_tokens, selected_placeholder_list):
            if ph == '__raw__':
                result.append(token)
            else:
                result.append('__PLACEHOLD__')
        return result[1:]

    def generate_edits_based_source(self, source_tokens, placeholder_list):
        source_tokens = ['__START__'] + source_tokens
        selected_placeholder_list = [i for i in placeholder_list if i != '__appended__']
        result = []
        for token, ph in zip(source_tokens, selected_placeholder_list):
            if ph == '__raw__':
                result.append(token)
            else:
                result.append('__PLACEHOLD__')
        return result[1:]

    def generate_correct_detail(self, text, source_tokens, pred_tokens, iter_label_idxs, iter_probs, iter_error_probs, config):
        placeholder_list = ['__raw__' for _ in range(1+len(source_tokens))] 
        for edit_probs, error_prob, label_idxs in zip(iter_probs, iter_error_probs, iter_label_idxs):
            # if error_prob < min_error_probability, convert the all the edit type to be '0' ($KEEP)
            if error_prob < config['min_error_probability']:
                label_idxs = [0] * len(label_idxs)
            else:
                # if edit_probability < min_probability, convert the edit type to be '0' ($KEEP)
                label_idxs = [idx if p>config['min_probability'] else 0 for (idx, p) in zip(label_idxs, edit_probs)]
            placeholder_list = self.overlay_edit(placeholder_list, label_idxs)

        edits_based_pred = self.generate_edits_based_pred(pred_tokens, placeholder_list)
        edits_based_source = self.generate_edits_based_source(source_tokens, placeholder_list)

        # print(len(source_tokens), source_tokens,'\n')
        # print(len(edits_based_source), edits_based_source,'\n')

        # print(len(pred_tokens), pred_tokens,'\n')
        # print(len(edits_based_pred), edits_based_pred,'\n')

        # 初始化
        correct_details = []
        source_start_idx = 0
        source_end_idx = 0
        pred_subgroup = []
        s = 0
        p = 0

        source_tokens_with_idx = add_tokens_idx(text, source_tokens)

        while s < len(edits_based_source)-1 and p < len(edits_based_source)-1:
            if edits_based_source[s] == edits_based_pred[p] and edits_based_pred[p]!='__PLACEHOLD__':
                s+=1
                p+=1
                continue

            while edits_based_source[s] == '__PLACEHOLD__':
                if source_end_idx == 0:
                    source_start_idx = source_tokens_with_idx[s][1]
                    source_end_idx = source_tokens_with_idx[s][2]
                else:
                    source_end_idx = source_tokens_with_idx[s][2]
                s += 1

            while edits_based_pred[p] == '__PLACEHOLD__':
                pred_subgroup.append(pred_tokens[p])
                p += 1

            if source_end_idx != 0 and pred_subgroup != []:
                source_substr = text[source_start_idx:source_end_idx] 
                pred_substr = ' '.join(pred_subgroup)
                logging.info("index[{}:{}] {} -> {}".format(source_start_idx, source_end_idx, source_substr, pred_substr))
                correct_details.append([source_substr, pred_substr, [source_start_idx, source_end_idx]])
                s+=1
                p+=1
                # 初始化
                source_start_idx = 0
                source_end_idx = 0
                pred_subgroup = []

            elif source_end_idx != 0 and pred_subgroup == []:
                if s!=0:
                    # (source_substr + source_tokens[s]) -> pred_tokens[p]
                    _, _, source_end_idx = source_tokens_with_idx[s]
                else:
                    # (source_tokens[s-1] + source_substr) -> pred_tokens[p]
                    _, source_start_idx, _ = source_tokens_with_idx[s-1]
                source_substr = text[source_start_idx:source_end_idx] 
                pred_substr = pred_tokens[p]
                logging.info("index[{}:{}] {} -> {}".format(source_start_idx, source_end_idx, source_substr, pred_substr))
                correct_details.append([source_substr, pred_substr, [source_start_idx, source_end_idx]])
                s+=1
                p+=1
                # 初始化
                source_start_idx = 0
                source_end_idx = 0
                pred_subgroup = []

            elif source_end_idx == 0 and pred_subgroup != []:
                if p!=0:
                    # source_tokens[s-1] -> pred_tokens[p-1-len(pred_subgroup)] + pred_subgroup
                    _, source_start_idx, source_end_idx = source_tokens_with_idx[s-1]
                    pred_substr = ' '.join( [pred_tokens[p-1-len(pred_subgroup)]] + pred_subgroup )
                else:
                    # source_tokens[s] -> pred_subgroup + pred_tokens[p]
                    _, source_start_idx, source_end_idx = source_tokens_with_idx[s]
                    pred_substr = ' '.join( pred_subgroup + [pred_tokens[p]])
                source_substr = text[source_start_idx:source_end_idx] 
                logging.info("index[{}:{}] {} -> {}".format(source_start_idx, source_end_idx, source_substr, pred_substr))
                correct_details.append([source_substr, pred_substr, [source_start_idx, source_end_idx]])
                s+=1
                p+=1
                # 初始化
                source_start_idx = 0
                source_end_idx = 0
                pred_subgroup = []
                
            else:
                s+=1
                p+=1
        
        # deduplication
        correct_details = [c for c in correct_details if c[0]!=c[1]]

        # check case_sentitive param
        if not config['case_sensitive']:
            correct_details = [c for c in correct_details if c[0].lower()!=c[1].lower()]
        return correct_details
