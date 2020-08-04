import scipy
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

def preprocess_embedding(z):
    """Pre-process embedding."""
    # Normalize coordinate system.
    # gmm = GaussianMixture(n_components=1, covariance_type='full')
    # gmm.fit(z)
    # mu = gmm.means_[0]
    # sigma = gmm.covariances_[0]
    gmm = GaussianMixture(n_components=1, covariance_type='spherical')
    gmm.fit(z)
    mu = gmm.means_[0]
    sigma = gmm.covariances_[0]
    z_norm = z - mu
    z_norm /= np.max(np.abs(z_norm))
    z_norm /= 2  # TODO experimenting with different scales here
    
    return z_norm


if __name__ == "__main__":
    n_dim = 30
    for modality in ['image', 'text']:
        # if modality == 'text':
            # emb = pickle.load(open('./data/real/raw/text.glove.50d.pickle','rb'))

        # same modality two embeddings test

        # emb_type = 'nn'
        emb_type = 'normal'

        if modality == 'text':
            # emb = pickle.load(open(f'./data/real/raw/{emb_type}/emb_text_dim_{n_dim}_2.pickle','rb'))
            emb = pickle.load(open(f'./data/real/raw/{emb_type}/emb_image_dim_{n_dim}_3.pickle','rb'))
        else:
            # emb = pickle.load(open(f'./data/real/raw/{emb_type}/emb_text_dim_{n_dim}_8.pickle','rb'))
            emb = pickle.load(open(f'./data/real/raw/{emb_type}/emb_image_dim_{n_dim}_9.pickle','rb'))

        print(list(emb.items())[0]) # check if its actually non-negative
        n_concept = len(emb.items())

        z = np.zeros((n_concept, n_dim))
        vocab = list(emb.keys())

        for i, word in enumerate(vocab):
            z[i] = emb[word]

        z = preprocess_embedding(z)

        with open(f'./data/real/{modality}.vec','w') as w:
            w.writelines(f'{n_concept} {n_dim}\n')
            for i, word in enumerate(vocab):
                w.writelines(f'{word} {" ".join([str(v) for v in z[i]])}\n')

        with open('./data/crosslingual/dictionaries/text-image.txt','w') as w:
            for word in vocab:
                w.writelines(f'{word}	{word}\n')

        with open('./data/crosslingual/dictionaries/image-text.txt','w') as w:
            for word in vocab:
                w.writelines(f'{word}	{word}\n')
    
    print('made data with dim ', n_dim)

    ### TESTING WITH SINGLE SYSTEM ROTATION ###
    # n_dim = 50
    # modality = 'text'
    # emb = pickle.load(open(f'./data/real/raw/emb_{modality}_dim_{n_dim}.pickle','rb'))
    # n_concept = len(emb.items())

    # z_text = np.zeros((n_concept, n_dim))
    # vocab = list(emb.keys())

    # for i, word in enumerate(vocab):
    #     z_text[i] = emb[word]

    # z_image = z_text + 0.01 * np.random.randn(n_concept, n_dim)

    # z_text = preprocess_embedding(z_text)
    # z_image = preprocess_embedding(z_image)

    # rot_mat = scipy.stats.special_ortho_group.rvs(z_text.shape[1])
    # z_image = np.matmul(z_image, rot_mat)

    
    # with open(f'./data/real/text.vec','w') as w:
    #     w.writelines(f'{n_concept} {n_dim}\n')
    #     for i, word in enumerate(vocab):
    #         w.writelines(f'{word} {" ".join([str(v) for v in z_text[i]])}\n')

    # with open(f'./data/real/image.vec','w') as w:
    #     w.writelines(f'{n_concept} {n_dim}\n')
    #     for i, word in enumerate(vocab):
    #         w.writelines(f'{word} {" ".join([str(v) for v in z_image[i]])}\n')

    
    
        
    