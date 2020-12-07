import os
import logging
from pathlib import Path


VOCAB_DIR = Path(__file__).resolve().parent.parent / "data"
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}


def get_verb_form_dicts():
    path_to_dict = os.path.join(VOCAB_DIR, "verb-form-vocab.txt")
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, tokens2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = tokens2
    return encode, decode


ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()


def get_target_sent_by_edits(source_tokens, edits):
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)


def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens

    target_line = " ".join(tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()


def convert_using_case(token, smart_action):
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


def convert_using_plural(token, smart_action):
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":
            return source_token
        # deal with case
        if transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        if transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        if transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        if transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token


def read_parallel_lines(fn1, fn2):
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    assert len(lines1) == len(lines2)
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    return out_lines1, out_lines2


def read_lines(fn, skip_strip=False):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]


def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)


def encode_verb_form(original_word, corrected_word):
    decoding_request = original_word + "_" + corrected_word
    decoding_response = ENCODE_VERB_DICT.get(decoding_request, "").strip()
    if original_word and decoding_response:
        answer = decoding_response
    else:
        answer = None
    return answer


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
    
def add_tokens_idx(text, source_tokens):
    # Get source tokens list with (start_idx, end_idx)
    # [['My', 0, 2],
    #  ['namme', 3, 8],
    #  ['is', 9, 11],
    #  ['are', 12, 15],
    #  ['was', 16, 19],
    #  ['Citao', 20, 25],
    #  ['.', 25, 26]]
    tokens_with_idx = []
    tmp_text = text[:]
    accumulated_lenght = 0
    for token in source_tokens:
        if not isinstance(token, str):
            token = str(token)
        word_start_idx = tmp_text.find(token) + accumulated_lenght
        word_end_idx = word_start_idx + len(token)
        tokens_with_idx.append([token, [word_start_idx, word_end_idx]])
        accumulated_lenght = word_end_idx
        tmp_text = text[word_end_idx:]
    return tokens_with_idx


def add_sents_idx(text, source_sentences):
    # Get source sentences list with (start_idx, end_idx)
    sents_with_idx = []
    tmp_text = text[:]
    accumulated_lenght = 0
    for sent in source_sentences:
        sent_start_idx = tmp_text.find(sent) + accumulated_lenght
        sent_end_idx = sent_start_idx + len(sent)
        sents_with_idx.append([sent, [sent_start_idx, sent_end_idx]])
        accumulated_lenght = sent_end_idx
        tmp_text = text[sent_end_idx:]
    return sents_with_idx


def minDistance(tokens1, tokens2):
    if len(tokens1) == 0:
        return len(tokens2)
    elif len(tokens2) == 0:
        return len(tokens1)
    M = len(tokens1)
    N = len(tokens2)
    output = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                output[i][j] = 0
            elif i == 0 and j != 0:
                output[i][j] = j
            elif i != 0 and j == 0:
                output[i][j] = i
            elif tokens1[i - 1] == tokens2[j - 1]:
                output[i][j] = output[i - 1][j - 1]
            else:
                output[i][j] = min(output[i - 1][j - 1] + 1, output[i - 1][j] + 1, output[i][j - 1] + 1)
    return output


def token_level_edits(tokens1, tokens2):
    dp = minDistance(tokens1,tokens2)
    m = len(dp)-1
    n = len(dp[0])-1
    operation = []
    spokenstr = []
    writtenstr = []
    edit = []

    while n>=0 or m>=0:
        if n and dp[m][n-1]+1 == dp[m][n]:
            edit.append(['', tokens2[n-1]])
            # print("insert [%s]" %(tokens2[n-1]))
            spokenstr.append("insert")
            writtenstr.append(tokens2[n-1])
            operation.append("NULLREF:"+tokens2[n-1])
            n -= 1
            continue
        if m and dp[m-1][n]+1 == dp[m][n]:
            edit.append([tokens1[m-1], ''])
            # print("delete [%s]" %(tokens1[m-1]))
            spokenstr.append(tokens1[m-1])
            writtenstr.append("delete")
            operation.append(tokens1[m-1]+":NULLHYP")
            m -= 1
            continue
        if dp[m-1][n-1]+1 == dp[m][n]:
            edit.append([tokens1[m-1], tokens2[n-1]])
            # print("replace [%s] [%s]" %(tokens1[m-1],tokens2[n-1]))
            spokenstr.append(tokens1[m - 1])
            writtenstr.append(tokens2[n-1])
            operation.append(tokens1[m - 1] + ":"+tokens2[n-1])
            n -= 1
            m -= 1
            continue
        if dp[m-1][n-1] == dp[m][n]:
            edit.append([tokens1[m-1], tokens1[m-1]])
            # print("keep")
            spokenstr.append(' ')
            writtenstr.append(' ')
            operation.append(tokens1[m-1])
        n -= 1
        m -= 1
    spokenstr = spokenstr[::-1]
    writtenstr = writtenstr[::-1]
    operation = operation[::-1]
    edit = edit[::-1]
    return spokenstr, writtenstr, operation, edit


def backward_merge_corrections(orig_text, corrections):
    """
    [['I', 'I', [0, 1]],
    ['went', 'went', [2, 6]],
    ['', 'to', [6, 6]],
    ['school', 'school', [7, 13]]]

    ->

    [['I', 'I', [0, 1]],
    ['went', 'went to', [2, 6]],
    ['school', 'school', [7, 13]]]
    """
    new_corrections = []
    i = len(corrections) - 1
    remains = []
    remains_len = 0

    while i > 0:
        r = corrections[i]
        if r[0] != r[1]:
            remains.append(r)
            remains_len += min(len(r[0]), len(r[1]))
        else:
            if remains:
                if remains_len > 0:
                    remains = remains[::-1]
                    start_idx = remains[0][2][0]
                    end_idx = remains[-1][2][1]
                    cor_sub_string = ' '.join([r[1] for r in remains])
                    orig_sub_string = orig_text[start_idx:end_idx]
                    new_corrections.append([orig_sub_string, cor_sub_string, [start_idx, end_idx]])
                    remains = []
                    remains_len = 0
                    new_corrections.append(r)
                else:
                    remains.append(r)
                    remains_len += min(len(r[0]), len(r[1]))
            else:
                new_corrections.append(r)
        i -= 1

    if remains!=[] and remains_len>0:
        remains = remains[::-1]
        start_idx = remains[0][2][0]
        end_idx = remains[-1][2][1]
        cor_sub_string = ' '.join([r[1] for r in remains])
        orig_sub_string = orig_text[start_idx:end_idx]
        new_corrections.append([orig_sub_string, cor_sub_string, [start_idx, end_idx]])

    new_corrections.append(corrections[0])
    new_corrections = new_corrections[::-1]
    return new_corrections


def forward_merge_corrections(orig_text, corrections):
    """
    [['I', 'I', [0, 1]],
    ['went', 'went', [2, 6]],
    ['', 'to', [6, 6]],
    ['school', 'school', [7, 13]]]

    ->

    [['I', 'I', [0, 1]],
    ['went', 'went', [2, 6]],
    ['school', 'to school', [7, 13]]]
    """
    new_corrections = []
    i = 0
    remains = []
    remains_len = 0

    while i < len(corrections)-1:
        r = corrections[i]
        if r[0] != r[1]:
            remains.append(r)
            remains_len += min(len(r[0]), len(r[1]))
        else:
            if remains:
                if remains_len > 0:
                    start_idx = remains[0][2][0]
                    end_idx = remains[-1][2][1]
                    cor_sub_string = ' '.join([r[1] for r in remains])
                    orig_sub_string = orig_text[start_idx:end_idx]
                    new_corrections.append([orig_sub_string, cor_sub_string, [start_idx, end_idx]])
                    remains = []
                    remains_len = 0
                    new_corrections.append(r)
                else:
                    remains.append(r)
                    remains_len += min(len(r[0]), len(r[1]))
            else:
                new_corrections.append(r)
        i += 1

    if remains!=[] and remains_len>0:
        start_idx = remains[0][2][0]
        end_idx = remains[-1][2][1]
        cor_sub_string = ' '.join([r[1] for r in remains])
        orig_sub_string = orig_text[start_idx:end_idx]
        new_corrections.append([orig_sub_string, cor_sub_string, [start_idx, end_idx]])

    new_corrections.append(corrections[-1])
    return new_corrections