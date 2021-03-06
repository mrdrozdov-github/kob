import gflags
import numpy as np
import json
import h5py
import os


UNK = "_"


LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}


class NLIObject(object):
    """
    Single item from an NLI dataset. Probably contains:

    - sentence1_tokens
    - sentence1_tokens_offset
    - sentence1_tokens_length
    - sentence1_transitions
    - sentence1_transitions_offset
    - sentence1_transitions_length
    - sentence2_tokens
    - sentence2_tokens_offset
    - sentence2_tokens_length
    - sentence2_transitions
    - sentence2_transitions_offset
    - sentence2_transitions_length
    - label
    - example_id
    """
    pass


def convert_binary_bracketing(parse):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                tokens.append(word.lower())
                transitions.append(0)
    return tokens, transitions


def load_data(path):
    print("Loading", path)
    examples = []
    tokens_offset = 0
    parse_offset = 0
    with open(path) as f:
        for ii, line in enumerate(f):
            line = line.strip()
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            example = NLIObject()
            example.label = LABEL_MAP[loaded_example["gold_label"]]
            example.example_id = ii
            (example.sentence1_tokens, example.sentence1_transitions) = convert_binary_bracketing(
                loaded_example["sentence1_binary_parse"])
            (example.sentence2_tokens, example.sentence2_transitions) = convert_binary_bracketing(
                loaded_example["sentence2_binary_parse"])

            examples.append(example)
    return examples


if __name__ == '__main__':
    datasets = [
        {
            "filename": os.path.expanduser('~/data/multinli_0.9/multinli_0.9_train.jsonl'),
            "h5name": 'train.h5'
        },
        {
            "filename": os.path.expanduser('~/data/multinli_0.9/multinli_0.9_dev_matched.jsonl'),
            "h5name": 'dev_matched.h5'
        },
    ]

    dt_vlen_str = h5py.special_dtype(vlen=str)

    for dset in datasets:
        filename = dset['filename']
        h5name = dset['h5name']
    
        data = load_data(filename)

        f = h5py.File(h5name, 'w')
        f.create_dataset("labels", data=np.asarray(map(lambda x: x.label, data), dtype=np.int32))
        f.create_dataset("example_id", data=np.asarray(map(lambda x: x.example_id, data), dtype=np.int32))
        f.create_dataset("sentence1_tokens", data=map(lambda x: json.dumps(x.sentence1_tokens), data), dtype=dt_vlen_str)
        f.create_dataset("sentence1_transitions", data=map(lambda x: json.dumps(x.sentence1_transitions), data), dtype=dt_vlen_str)
        f.create_dataset("sentence2_tokens", data=map(lambda x: json.dumps(x.sentence2_tokens), data), dtype=dt_vlen_str)
        f.create_dataset("sentence2_transitions", data=map(lambda x: json.dumps(x.sentence2_transitions), data), dtype=dt_vlen_str)
        f.close()
