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
            if coder != args.id: continue # Ignore other coders
            span = edit[0].split()[1:] # Ignore "A "
            start = int(span[0])
            end = int(span[1])
            cor = edit[2].split()
            cor_sent[start+offset:end+offset] = cor
            offset = offset-(end-start)+len(cor)
        target_f.write(" ".join(cor_sent)+"\n")
        source_f.write(" ".join(sent[0].split()[1:])+"\n")
    print('Processed {}'.format(m2_file))
    return None


if __name__ == "__main__":
    m2_source_list = [
        "/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/fce/m2",
        "/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/lang8",
        "/home/citao/Edison_workspace/Edison_tmp_data/autocorrect/bea2019/wi+locness/m2"
    ]

    save_path = "/home/citao/github/gector/dataset/"
    for ms in m2_source_list:
        m2_files = glob.glob(os.path.join(ms, "./*.m2"))
        for m2_file in m2_files:
            _index = m2_file.split('/')[-1]
            source_file = os.path.join(save_path, _index.replace(".m2", ".source"))
            target_file = os.path.join(save_path, _index.replace(".m2", ".source"))
            print(m2_file)
            print(source_file)
            print(target_file)
            print()




