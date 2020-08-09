import scipy
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
    # z_norm = (z - mu) / sigma
    z_norm = z - mu
    max_val = np.max(np.abs(z_norm))
    z_norm = z_norm / max_val
    z_norm = z_norm / 2  # TODO experimenting with different scales here
    # z_norm + 1  # TODO
    return z_norm


if __name__ == "__main__":
    n_concept = 200
    n_dim = 2

    z_0 = np.random.randn(n_concept, n_dim)
    z_1 = z_0 + 0.01 * np.random.randn(n_concept, n_dim)

    z_0 = preprocess_embedding(z_0)
    z_1 = preprocess_embedding(z_1)

    rot_mat = scipy.stats.special_ortho_group.rvs(z_0.shape[1])
    z_1 = np.matmul(z_1, rot_mat)

    idx_rand = np.random.permutation(n_concept)
    z_1_shuffle = z_1[idx_rand, :]
    # Determine correct mapping.
    y_idx_map = np.argsort(idx_rand)
    np.testing.assert_array_equal(z_1, z_1_shuffle[y_idx_map, :])

    with open('./data/synthetic/z_0.vec','w') as w:
        for i in range(n_concept):
            w.writelines(f'{i} {" ".join([str(v) for v in z_0[i]])}\n')

    with open('./data/synthetic/z_1.vec','w') as w:
        for i in range(n_concept):
            w.writelines(f'{y_idx_map[i]} {" ".join([str(v) for v in z_1_shuffle[y_idx_map[i]]])}\n')
    
    with open('./data/crosslingual/dictionaries/z_0-z_1.txt','w') as w:
        for i in range(n_concept):
            w.writelines(f'{i} {y_idx_map[i]}\n')

    with open('./data/crosslingual/dictionaries/z_1-z_0.txt','w') as w:
        for i in range(n_concept):
            w.writelines(f'{y_idx_map[i]} {i}\n')