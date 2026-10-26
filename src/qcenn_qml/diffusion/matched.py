"""Raw quantum corpus memory, classical references, and paired readout noise.

There are no fitted classical conditional probabilities in the quantum gates.
The classical conditional-table baseline is evaluated separately for comparison.
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import re
import numpy as np

from .core import dataset as toy_dataset, VOCAB


TEXT="""the cat sat on the mat.
the dog sat on the mat.
the cat ran to the door.
the dog ran to the door.
the bird sat in the tree.
the cat slept in the room.
the dog slept in the room.
the bird flew to the tree.
the child sat on the mat.
the child ran to the door.
the cat sat in the room.
the dog sat in the room.
the child slept in the room.
the bird sat on the mat.
the cat ran to the tree.
the dog ran to the tree.
"""


def corpus(corpus_kind="text",corpus_file=None,seed=42, *, max_train_records=256, max_vocabulary=64):
    if corpus_kind=="toy" and corpus_file is None:
        train,test=toy_dataset(seed)
        return list(VOCAB),3,train,test,{"source":"legacy_three_token_toy","split":"seeded_records"}
    text=Path(corpus_file).read_text(encoding='utf8') if corpus_file else TEXT
    units=[re.findall(r"[a-z]+",s.lower()) for s in re.split(r"[.!?\n]+",text)]
    units=[u for u in units if len(u)>=3]
    if len(units)<4:
        raise ValueError("Provide at least four sentences of three or more words")
    if not 1 <= max_train_records <= 256 or not 4 <= max_vocabulary <= 64:
        raise ValueError('Memory budgets require 1..256 records and 4..64 vocabulary entries')
    # Identical sentences stay in the same partition even when repeated in a file.
    groups={}
    for index,unit in enumerate(units):
        groups.setdefault(tuple(unit),[]).append(index)
    if len(groups)<4:
        raise ValueError('Provide at least four distinct sentences')
    keys=list(groups)
    order=np.random.default_rng(seed).permutation(len(keys))
    boundary=max(1,min(len(keys)-1,int(.75*len(keys))))
    train_ids=[index for group in order[:boundary] for index in groups[keys[group]]]
    heldout_ids=[index for group in order[boundary:] for index in groups[keys[group]]]
    training=[units[i] for i in train_ids]
    heldout=[units[i] for i in heldout_ids]
    frequencies=Counter(w for sentence in training for w in sentence)
    retained=sorted(frequencies,key=lambda word:(-frequencies[word],word))[:max_vocabulary-2]
    vocab=['<unk>','[MASK]']+sorted(retained)
    mapping={w:i for i,w in enumerate(vocab)}
    def windows(sentences):
        return [tuple(mapping.get(w,0) for w in sentence[i:i+3])
                for sentence in sentences for i in range(len(sentence)-2)]
    all_train,test=windows(training),windows(heldout)
    selected=np.random.default_rng(seed+701).permutation(len(all_train))[:max_train_records]
    train=[all_train[int(index)] for index in selected]
    heldout_windows=[{'sentence_id':index,'words':sentence[i:i+3],
        'target_in_vocabulary':sentence[i+1] in mapping,
        'context_in_vocabulary':sentence[i] in mapping and sentence[i+2] in mapping}
        for index,sentence in zip(heldout_ids,heldout,strict=True) for i in range(len(sentence)-2)]
    return vocab,1,train,test,{"source":str(corpus_file) if corpus_file else "bundled_small_English_text",
        "corpus_sha256":hashlib.sha256(text.encode('utf8')).hexdigest(),
        "split":"whole_sentences_before_windowing","train_sentence_ids":train_ids,
        "heldout_sentence_ids":heldout_ids,"heldout_vocabulary_from_training_only":True,
        'duplicate_sentences_grouped_before_split':True,'distinct_sentence_groups':len(groups),
        'training_windows_before_memory_cap':len(all_train),'selected_training_window_indices':selected.tolist(),
        'memory_cap_selected_without_heldout_labels':True,'heldout_windows':heldout_windows,
        'vocabulary_limit':max_vocabulary,'unknown_targets_receive_no_accuracy_credit':True}


def execute_benchmark(*args, **kwargs):
    from .evaluation import execute_benchmark as evaluate
    return evaluate(*args, **kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots',type=int,default=1024)
    parser.add_argument('--beta',type=float,default=1.)
    parser.add_argument('--eta',type=float,default=.65)
    parser.add_argument('--steps',type=int,nargs='+',default=[0,1,3])
    parser.add_argument('--examples',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--readout-error',type=float,default=.05)
    parser.add_argument('--readout-errors',type=float,nargs='+',help='Additional paired readout-error levels; noiseless always runs')
    parser.add_argument('--measurement-seed',type=int)
    parser.add_argument('--corpus-kind',choices=['toy','text'],default='text')
    parser.add_argument('--corpus-file')
    parser.add_argument('--amplify',type=int,default=1)
    parser.add_argument('--max-train-records',type=int,default=256)
    parser.add_argument('--max-vocabulary',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output-dir',default='results/diffusion-matched')
    return execute_benchmark(**vars(parser.parse_args(argv)))


if __name__=='__main__':
    main()
