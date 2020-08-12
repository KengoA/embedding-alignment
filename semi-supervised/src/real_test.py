# -*- coding: utf-8 -*-
# Copyright 2020 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Use synthetic data to evaluate alignment algorithm."""

import copy
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Add
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
# import tensorflow_graphics as tfg
tfd = tfp.distributions
kl = tfd.kullback_leibler

import embeddings
import aligners
import utils

import utils

def main(fp_repo):
    """Run script."""
    # Settings
    tf.keras.backend.set_floatx('float64')
    fp_intersect = fp_repo / Path('python', 'assets', 'intersect')    

    # Settings
    tf.keras.backend.set_floatx('float64')

    # Load real embeddings
    z_0, z_1, vocab = embeddings.load_text_and_image()

    # Pre-process embeddings.
    z_0 = utils.preprocess_embedding(z_0)
    z_1 = utils.preprocess_embedding(z_1)

    # TEST ON ONE SYSTEM WITH ROTATION
    # np.random.seed(3123)
    # rot_mat = scipy.stats.special_ortho_group.rvs(z_0.shape[1])
    # noise = 0.01 * np.random.randn(z_0.shape[0], z_0.shape[1])
    # z_0 = np.matmul(z_1, rot_mat) + noise

    n_concept = z_0.shape[0]
    n_dim = z_0.shape[1]
    fully_unsupervised = False
    n_restart = 1

    if fully_unsupervised:
        sup_size = 0.00
        max_epoch = 100
        log_per_epoch = False

    else:
        sup_size = 0.05
        max_epoch = 1000
        log_per_epoch = True

    logger = utils.setup_logger(name_logfile=f'real_sup_size_{sup_size}.log', logs_dir='./logs/', also_stdout=True)

    # Determine ceiling performance.
    # template = 'Ceiling Accuracy 1: {0:.2f} 5: {1:.2f} 10: {2:.2f} Half: {3:.2f}\n'
    # acc_1, acc_5, acc_10, acc_half = utils.mapping_accuracy(z_0, z_1)
    # print(template.format(acc_1, acc_5, acc_10, acc_half))
    # logger.info(template.format(acc_1, acc_5, acc_10, acc_half))

    # Shuffle second embedding, but keep track of correct mapping.
    idx_rand = np.random.permutation(n_concept)
    z_1_shuffle = z_1[idx_rand, :]
    # Determine correct mapping.
    y_idx_map = np.argsort(idx_rand)
    # Verify mapping to be safe.
    np.testing.assert_array_equal(z_1, z_1_shuffle[y_idx_map, :])

    # Align embeddings using alignment algorithm 'bdr_0_0_1'.
    aligners.bdr_0_0_1(z_0, z_1_shuffle, logger, log_per_epoch, y_idx_map, n_dim, max_epoch, sup_size, n_restart)

if __name__ == "__main__":
    # Specify the path to the repository folder.
    fp_repo = Path.home() / Path('projects', 'unsupervised-alignment-team')
    main(fp_repo)