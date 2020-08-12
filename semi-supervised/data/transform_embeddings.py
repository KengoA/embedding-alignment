import pickle
import numpy as np

if __name__ == '__main__':
    emb_dir = '../glove/data/embeddings'
    n_dim = 30
    emb_type = 'nn'
    text_path = f'{emb_dir}/{emb_type}/emb_image_dim_{n_dim}_3.pickle'
    # text_path = f'{emb_dir}/{emb_type}/emb_text_dim_{n_dim}_1.pickle'

    # image_path = f"{emb_dir}/emb_image_dim_{n_dim}_0.pickle"
    image_path = f'{emb_dir}/{emb_type}/emb_image_dim_{n_dim}_1.pickle'
    # image_path = f'{emb_dir}/{emb_type}/emb_text_dim_{n_dim}_1.pickle'

    # text_path = f'{emb_dir}/emb_text_dim_{n_dim}_0.pickle'
    # image_path = f'{emb_dir}/emb_text_dim_{n_dim}_1.pickle'

    emb_text = pickle.load(open(text_path,'rb'))
    emb_image = pickle.load(open(image_path,'rb'))

    vocab = []
    z_text = []
    z_image = []

    for key in emb_text.keys():
        if key in emb_image.keys():
            vocab.append(key)
            z_text.append(emb_text[key])
            z_image.append(emb_image[key])
    
    data = {'z_0': np.array(z_text), 'z_1':np.array(z_image), 'vocab': np.array(vocab)}
    with open(f'./data/preprocessed/text-image.pickle','wb') as w:
        pickle.dump(data, w)