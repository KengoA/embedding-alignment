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

import numpy as np
import scipy.stats
import tensorflow as tf

import embeddings
import aligners
import utils


def main(fp_repo):
    """Run script."""
    # Settings
    tf.keras.backend.set_floatx('float64')
    n_concept = 200

    noise = 0.01
    n_dim = 2
    fully_unsupervised = False

    if fully_unsupervised:
        n_restart = 1000
        sup_size = 0.00
        max_epoch = 30
        log_per_epoch = False

    else:
        n_restart = 1
        sup_size = 0.5
        max_epoch = 100
        log_per_epoch = True

    logger = utils.setup_logger(name_logfile=f'n_concept_{n_concept}_noise_{noise}_n_dim_{n_dim}_sup_size_{sup_size}_max_epoch_{max_epoch}.log', logs_dir='../logs/', also_stdout=True)

    # Load synthetic embeddings.
    z_0, z_1 = embeddings.noisy_gaussian(n_concept, noise=noise, n_dim=n_dim)

    # Pre-process embeddings.
    z_0 = utils.preprocess_embedding(z_0)
    z_1 = utils.preprocess_embedding(z_1)

    # Determine ceiling performance.
    template = 'Ceiling Accuracy 1: {0:.2f} 5: {1:.2f} 10: {2:.2f} Half: {3:.2f}\n'
    acc_1, acc_5, acc_10, acc_half = utils.mapping_accuracy(z_0, z_1)
    # print(template.format(acc_1, acc_5, acc_10, acc_half))
    logger.info(template.format(acc_1, acc_5, acc_10, acc_half))

    # Add random rotation to the second embedding.
    np.random.seed(3123)
    rot_mat = scipy.stats.special_ortho_group.rvs(z_0.shape[1])
    z_1 = np.matmul(z_1, rot_mat)

    # Shuffle second embedding, but keep track of correct mapping.
    n_concept = z_0.shape[0]
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
