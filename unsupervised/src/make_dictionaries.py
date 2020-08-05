import pickle

if __name__=='__main__':
    emb = pickle.load(open(f'./data/real/raw/nn/emb_text_dim_30_0.pickle','rb'))
    vocab = list(emb.keys())
    # dictionaries, run once
    with open('./data/crosslingual/dictionaries/src-tgt.txt','w') as w:
        for word in vocab:
            w.writelines(f'{word}	{word}\n')

    with open('./data/crosslingual/dictionaries/src-tgt.txt','w') as w:
        for word in vocab:
            w.writelines(f'{word}	{word}\n')