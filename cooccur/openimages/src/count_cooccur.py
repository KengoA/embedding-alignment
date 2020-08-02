import time
import gc
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    df_vocab_mid = pd.read_csv('../../vocab/intersect_vocab.csv')
    
    VOCAB = df_vocab_mid['name'].values.tolist()
    MID = df_vocab_mid['mid'].values.tolist()

    mid2name = {}

    for mid, name in zip(MID, VOCAB):
        mid2name[mid] = name

    df = pd.read_csv('../data/preorocessed/open-images.csv')

    cooccur_mid = {concept: {context: 0 for context in MID} for concept in MID}
    cooccur_name = {concept: {context: 0 for context in VOCAB} for concept in VOCAB}

    images = df.groupby('ImageID')['LabelName'].apply(pd.Series.tolist).tolist()

    del df
    gc.collect()

    image_classes = []
    count = 0

    for labels in tqdm(images):
        image_classes.append(len(labels))
        for i in range(len(labels)):
            for j in range(len(labels)):
                if (labels[i] in MID) and (labels[j] in MID):
                    if i != j:
                        cooccur_mid[labels[i]][labels[j]] += 1 
                        cooccur_name[mid2name[labels[i]]][mid2name[labels[j]]] += 1 
                        count += 1

    print("image class -- mean: {0:.2f} (std: {1:.2f}) max: {2:.2f}".format(np.mean(image_classes), 
                                                                            np.std(image_classes),
                                                                            np.max(image_classes)))
    print("total cooccurrence counted: {}".format(count))

    save_dir = '../../../glove/data/cooccur/'
    with open(save_dir+'cooccur_image_name.json', 'w') as w:
        json.dump(cooccur_name, w)

    df = pd.read_json(save_dir+'cooccur_image_name.json')
    df.to_csv(save_dir+'cooccur_image_name.csv')

    with open(save_dir+'cooccur_image_mid.json', 'w') as w:
        json.dump(cooccur_mid, w)

    df = pd.read_json(save_dir+'cooccur_image_mid.json')
    df.to_csv(save_dir+'cooccur_image_mid.csv')
                

