#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize(exp_dir):
    # Append results
    results = {'modality':[], 'emb_type':[], 'idx_src':[], 'idx_tgt':[], 'epoch':[], 'loss':[], 'acc':[]}

    diff = {'text-nn':[], 'text-normal':[], 'image-nn':[], 'image-normal':[]}

    for modality in ['text', 'image']:
        for emb_type in ['nn', 'normal']:
            for idx_src in range(0,4):
                for idx_tgt in range(0,4):
                    if idx_src != idx_tgt:
                        log_dir = os.path.join(exp_dir, f'{emb_type}_{modality}_{idx_src}-{emb_type}_{modality}_{idx_tgt}')
                        with open(os.path.join(log_dir, 'eval_acc.log'), 'r') as r:
                            acc_list = []
                            for line in r.readlines():
                                acc = float(line.split('Accuracy: ')[1][:-2])
                                acc_list.append(acc)
                            diff[f'{modality}-{emb_type}'].append(np.mean(acc_list))

    results = {'modality':[], 'emb_type':[], 'idx_src':[], 'idx_tgt':[], 'epoch':[], 'loss':[], 'acc':[]}

    for modality in ['text', 'image']:
        for emb_type in ['nn', 'normal']:
            for idx_src in range(0,4):
                for idx_tgt in range(0,4):
                    if idx_src != idx_tgt:
                        log_dir = os.path.join(exp_dir, f'{emb_type}_{modality}_{idx_src}-{emb_type}_{modality}_{idx_tgt}')
                        with open(os.path.join(log_dir, 'logger.log'), 'r') as r:
                            prev = None
                            for line in r.readlines():
                                if 'epoch' in line:
                                    epoch = float(line.split()[3].split(':')[1][:-1])           
                                    loss = float(line.split()[5].split(':')[1])
                                    acc = float(line.split()[-1])
                                    
                                    if prev != acc:
                                        results['modality'].append(modality)
                                        results['emb_type'].append(emb_type)
                                        results['idx_src'].append(idx_src)
                                        results['idx_tgt'].append(idx_tgt)
                                        results['epoch'].append(epoch)
                                        results['loss'].append(loss)
                                        results['acc'].append(acc)
                                        prev = acc

    df = pd.DataFrame(results)

    for modality in ['image','text']:
        for emb_type in ['nn', 'normal']:
            plt.figure(figsize=(16,10))
            for idx_src in range(0,4):
                for idx_tgt in range(0,4):
                    if idx_src != idx_tgt:
                        df_plot = df.loc[(df['idx_src']==idx_src) & (df['idx_tgt']==idx_tgt) & (df['modality']==modality) & (df['emb_type']==emb_type)]
                        plt.plot(df_plot['epoch'], df_plot['acc'])


            title_prefix = 'Image-to-Image' if modality=='image' else 'Text-to-Text'
            title_suffix = '(Non-Negative)' if emb_type=='nn' else '(No Constraint)'
            title_main = 'Alignment Accuracy'

            plt.title(f'{title_prefix} {title_main} {title_suffix}', fontsize=16, pad=20)

            plt.text(500,1.02,'     WGAN Loss Phase \n(epochs 0-1500) ',fontsize=14)
            plt.text(2700,1.02,'     Sinkhorn Loss Phase \n(epochs 1501-4500)',fontsize=14)
            plt.axvline(x=1500, ls='--')
            plt.xticks(np.arange(0, 4650, 150))
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('Alignment Accuracy', fontsize=14)
            plt.xlim(0,4500)
            plt.ylim(0,1.12)
            plt.yticks(np.arange(0, 1.1, 0.1))
            
            # statistical info box
            N = 12
            mu = np.mean(diff[f'{modality}-{emb_type}'])/100
            sigma = np.std(diff[f'{modality}-{emb_type}'])/100       
            
            textstr = '\n'.join((
                r'$N=%.1d$' % (N, ),
                r'$mean=%.4f$' % (mu, ),
                r'$std=%.4f$' % (sigma, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            plt.text(4000, 1.09, textstr, fontsize=14, verticalalignment='top', bbox=props)
            plt.savefig(f'../assets/{title_prefix} {title_main} {title_suffix}.pdf')
            plt.show()


if __name__=='__main__':
    exp_dir = '../exp/'
    visualize(exp_dir)




