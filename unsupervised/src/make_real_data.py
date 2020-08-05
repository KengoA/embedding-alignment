import argparse
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
    parser = argparse.ArgumentParser(description='make real dataset from embeddings')
    parser.add_argument('--n_dim',
        type=int, help='number of dimensions', default=30)
    parser.add_argument('--n_concept',
        type=int, help='number of concepts', default=429)
    parser.add_argument('--emb_type',
        type=str, help='type of embeddings, either normal or nn (nonnegative)', default='nn')
    parser.add_argument('--modality_src',
        type=str, help='modality of source embeddings', default='text')
    parser.add_argument('--modality_tgt',
        type=str, help='modality of target embeddings', default='image')
    parser.add_argument('--idx_src',
        type=int, help='index of source embeddings', default=0)
    parser.add_argument('--idx_tgt',
        type=int, help='index of target embeddings', default=0)

    args = parser.parse_args()

    # if args.modality == 'text':
        # emb = pickle.load(open('./data/real/raw/text.glove.50d.pickle','rb'))

    # same modality two embeddings test

    emb_src = pickle.load(open(f'./data/real/raw/{args.emb_type}/emb_{args.modality_src}_dim_{args.n_dim}_{args.idx_src}.pickle','rb'))
    emb_tgt = pickle.load(open(f'./data/real/raw/{args.emb_type}/emb_{args.modality_tgt}_dim_{args.n_dim}_{args.idx_tgt}.pickle','rb'))
    
    # print('src',list(emb_src.items())[0]) # check if its actually non-negative
    # print('tgt',list(emb_tgt.items())[0]) # check if its actually non-negative
    
    def save_vectors(emb, is_src=True):
        z = np.zeros((args.n_concept, args.n_dim))
        vocab = list(emb.keys())

        for i, word in enumerate(vocab):
            z[i] = emb[word]

        z = preprocess_embedding(z)

        emb_name = f"{args.emb_type}_{args.modality_src}_{args.idx_src}" if is_src else f"{args.emb_type}_{args.modality_tgt}_{args.idx_tgt}"

        with open(f"./data/real/{emb_name}.vec",'w') as w:
            w.writelines(f"{args.n_concept} {args.n_dim}\n")
            for i, word in enumerate(vocab):
                w.writelines(f'{word} {" ".join([str(v) for v in z[i]])}\n')
        
        return None
                
    save_vectors(emb_src, is_src=True)
    save_vectors(emb_tgt, is_src=False)
        
    print(f'made data for {args.emb_type} emb with {args.n_dim} dims for {args.modality_src} {args.idx_src} and {args.modality_tgt} {args.idx_tgt}')


    

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

    
    
        
    