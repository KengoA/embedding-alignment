import glob
import pandas as pd
from joblib import Parallel, delayed

import cooccur

if __name__ == "__main__":
    paths = sorted(glob.glob('../data/preprocessed/*'))
    vocab = pd.read_csv('../../vocab/intersect_vocab.csv')['name'].values.tolist()

    # cooccur_cy.count_cooccur(paths[0], vocab)

    # for path in paths[0]:
    #     cooccur_cy.count_cooccur(path, vocab)

    Parallel(n_jobs=-1, verbose=2)(delayed(cooccur.count_cooccur)(path, vocab) for path in paths)
