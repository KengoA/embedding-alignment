import json
import glob
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__": 
    VOCAB = pd.read_csv('../../vocab/intersect_vocab.csv')['name'].values.tolist()
    MID = pd.read_csv('../../vocab/intersect_vocab.csv')['mid'].values.tolist()
    
    name2mid = {}

    for mid, name in zip(MID, VOCAB):
        name2mid[name] = mid

    M = {concept: {context: 0 for context in VOCAB} for concept in VOCAB}
    M_mid = {concept_mid: {context_mid: 0 for context_mid in MID} for concept_mid in MID}

    for path in tqdm(glob.glob('../data/intermediate/*')):
        with open(path, 'r') as r:
            cooccur = json.load(r)
            for concept in cooccur.keys():
                for context in cooccur[concept].keys():
                    M[concept][context] += cooccur[concept][context]

    save_dir = '../../../glove/data/cooccur/'
    with open(save_dir+'cooccur_text_name.json', 'w') as w:
        json.dump(M, w)

    df = pd.read_json(save_dir+'cooccur_text_name.json')
    df.to_csv(save_dir+'cooccur_text_name.csv')

    for concept in M.keys():
        for context in M[concept].keys():
            M_mid[name2mid[concept]][name2mid[context]] = M[concept][context]

    with open(save_dir+'cooccur_text_mid.json', 'w') as w:
        json.dump(M_mid, w)

    df = pd.read_json(save_dir+'cooccur_text_mid.json')
    df.to_csv(save_dir+'cooccur_text_mid.csv')
