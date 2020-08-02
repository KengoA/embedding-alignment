import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob

from metrics import calc_cosine_sim, calc_eucl_sim

if __name__=='__main__':
    ROOT_DIR = '/content/gdrive/My Drive/0_MSc_Project/'
    vocab = pd.read_csv(ROOT_DIR+'intersect_vocab.csv')['name'].values

    # paths = glob(ROOT_DIR+'word_similarity/*')
    # for path in paths:
    #     df = pd.read_csv(path)
    #     df['word1'] = df['word1'].str.lower()
    #     df['word2'] = df['word2'].str.lower()
    #     count = 0
    #     for i in range(df.shape[0]):
    #         if (df.loc[i,'word1'] in vocab) and (df.loc[i,'word2'] in vocab):
    #             count += 1
    #     print('{} {}/{}'.format(path.split('/')[-1], count,df.shape[0]))


    # df = pd.read_csv(ROOT_DIR+'word_similarity/wordsim353-sim.csv')
    # df = pd.read_csv(ROOT_DIR+'word_similarity/semeval17.csv')
    df = pd.read_csv(ROOT_DIR+'word_similarity/mturk-771.csv')
    # df = pd.read_csv(ROOT_DIR+'word_similarity/simlex999.csv')

    indices = []

    for i in range(df.shape[0]):
        if (df.loc[i,'word1'] in vocab) and (df.loc[i,'word2'] in vocab):
            indices.append(i)

    df_sim = df.loc[indices,['word1','word2','similarity']].reset_index(drop=True)
    df_sim['similarity'] = df_sim['similarity']/df['similarity'].max()

    emb_sim = []

    for i in range(df_sim.shape[0]):
        word1 = df_sim.loc[i,'word1']
        word2 = df_sim.loc[i,'word2']
        word1_emb = emb[dataset._name_to_id[word1],:]
        word2_emb = emb[dataset._name_to_id[word2],:]
        # emb_sim.append(calc_cosine_sim(word1_emb, word2_emb))
        emb_sim.append(calc_eucl_sim(word1_emb, word2_emb))

    df_sim['emb_sim'] = emb_sim
    print(df_sim.corr())
    df_sim