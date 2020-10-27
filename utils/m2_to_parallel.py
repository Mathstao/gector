import os
import glob

def parse_m2_file(m2_file, source_file, target_file, annotator_id):
    m2 = open(m2_file, 'r', encoding='utf-8').read().strip().split("\n\n")

    # Do not apply edits with these error types
    skip = {"noop", "UNK", "Um"}

    source_f = open(source_file, 'w', encoding='utf-8')
    target_f = open(target_file, 'w', encoding='utf-8')

    for sent in m2:
        sent = sent.split("\n")
        cor_sent = sent[0].split()[1:] # Ignore "S "
        edits = sent[1:]
        offset = 0
        for edit in edits:
            edit = edit.split("|||")
            if edit[1] in skip: continue # Ignore certain edits
            coder = int(edit[-1])
            if coder != annotator_id: continue # Ignore other coders
            span = edit[0].split()[1:] # Ignore "A "
            start = int(span[0])
            end = int(span[1])
            cor = edit[2].split()
            cor_sent[start+offset:end+offset] = cor
            offset = offset-(end-start)+len(cor)
        target_f.write(" ".join(cor_sent)+"\n")
        source_f.write(" ".join(sent[0].split()[1:])+"\n")
    return None


if __name__ == "__main__":
    m2_source_list = [
        "/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/fce/m2",
        "/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/lang8",
        "/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/wi+locness/m2"
    ]

    save_path = "/home/citao/github/gector/dataset/"
    for ms in m2_source_list:
        m2_files = glob.glob(os.path.join(ms, "*.m2"))
        for m2_file in m2_files:
            if "lang8" in m2_file:
                annotators = [0, 1, 2, 3, 4]
            else:
                annotators = [0]
            for annotator_id in annotators:
                _index = m2_file.split('/')[-1]
                _index = _index.replace(".m2", ".{}.m2".format(annotator_id))
                if "wi+locness" in m2_file:
                    _index = "wil."+_index
                source_file = os.path.join(save_path, _index.replace(".m2", ".source"))
                target_file = os.path.join(save_path, _index.replace(".m2", ".target"))
                # parse_m2_file(m2_file, source_file, target_file, annotator_id)
                print(m2_file)
                print(source_file)
                print(target_file)
                print()
    # merge
    for name in ['fce', 'lang8', 'wil']:
        for part in ['train', 'dev', 'test']:
            for dtype in ['source', 'target']:
                merged_list = glob.glob(os.path.join(save_path, "{}*{}*{}".format(name, part, dtype)))
                if merged_list != []:
                    dest = os.path.join(save_path, '{}.{}.{}'.format(name, part, dtype))
                    cmd = 'cat {} > {}'.format(' '.join(merged_list), dest)
                    print(cmd)
                    os.popen(cmd)





