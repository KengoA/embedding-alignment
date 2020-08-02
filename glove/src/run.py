if __name__ == "__main__":
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = 'cuda:0'
    else:
        device = 'cpu'

    ROOT_DIR = '/content/gdrive/My Drive/0_MSc_Project/'


    for emb_type in ['normal','nn']:
        for emb_i in range(10):
            for modality in ['text','image']:
                cooccur = json.load(open(ROOT_DIR+f'../data/cooccur_{modality}_name.json', 'r'))
                data = {'id_i':[], 'id_j':[], 'X_ij':[]}

                for center in cooccur.keys():
                    for context in cooccur[center].keys():
                        if cooccur[center][context] > 0:
                            data['id_i'].append(center)
                            data['id_j'].append(context)
                            data['X_ij'].append(cooccur[center][context])

                with open(ROOT_DIR+f'cooccur_{modality}.json','w') as w:
                    json.dump(data, w)

                batch_size = 256

                dataset = Dataset(ROOT_DIR+f'cooccur_{modality}.json', random_id=False)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

                #################

                epochs = 300

                embedding_dim = 30
                l1_lambda = 0.00
                vocab_len = len(dataset._name_to_id.keys())

                glove = GloVe(vocab_len, embedding_dim).to(device)
                optimizer = optim.Adagrad(glove.parameters(), lr=0.075)

                loss_hist = []
                print(f'Training GloVe Embeddings for {modality} with dimensions {embedding_dim}')

                for epoch in tqdm(range(epochs)):
                    batch_hist_total, batch_hist_glove, batch_hist_l1_wi, batch_hist_l1_wj = [], [], [], []
                    for batch_i, (id_i, id_j, X_ij) in enumerate(dataloader):
                        optimizer.zero_grad()
                        
                        wi, wj, bi, bj = glove(id_i, id_j)

                        inputs = torch.einsum('ij, ij -> i', wi, wj) + bi + bj
                        targets = torch.log(X_ij)
                        weights = f(X_ij)
                        
                        glove_loss = wmse(inputs, targets, weights)

                        l1_wi = l1_lambda*torch.mean(torch.abs(wi))
                        l1_wj = l1_lambda*torch.mean(torch.abs(wj))
                        
                        total_loss = sum([glove_loss + l1_wi + l1_wj])

                        total_loss.backward()
                        optimizer.step()

                        if emb_type == 'nn':
                            # Non-negativity constraint
                            glove.wi.weight.data.clamp_(0)
                            glove.wj.weight.data.clamp_(0)

                        batch_hist_total.append(total_loss.item())
                        batch_hist_glove.append(glove_loss.item())
                        batch_hist_l1_wi.append(l1_wi.item())
                        batch_hist_l1_wj.append(l1_wj.item())

                        loss_hist.append(total_loss.item())
                        
                    print('epoch:{0:1d}/{1:1d} loss:{2:.4f} glove: {3:.4f} +-({4:.4f}) l1 wi: {5:.4f} wj: {6:.4f}'.format(
                        epoch+1, epochs, np.mean(batch_hist_total), np.mean(batch_hist_glove), np.std(batch_hist_glove), np.mean(batch_hist_l1_wi), np.mean(batch_hist_l1_wj)
                    ))

                plt.plot(loss_hist)
                plt.show()

                ##################

                wi = glove.wi.weight
                wj = glove.wj.weight
                emb = (wi + wj).cpu().detach().numpy()

                tsne = TSNE(metric='cosine')
                embed_tsne = tsne.fit_transform(emb)

                fig, ax = plt.subplots(figsize=(14, 14))

                indices = np.random.choice(range(vocab_len), 429, replace=False)

                for idx in indices:
                    plt.scatter(*embed_tsne[idx,:])
                    plt.annotate(dataset._id_to_name[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

                plt.show()

                # # Save as a text file
                # with open(ROOT_DIR+f'embeddings/emb_{modality}_dim_{embedding_dim}_{emb_i}.txt', 'w') as w:
                #     for idx in dataset._id_to_name.keys():
                #         w.writelines(dataset._id_to_name[idx]+' '+' '.join([str(v) for v in emb[idx]])+'\n')


                # Save as a pickle file
                with open(ROOT_DIR+f'embeddings/{emb_type}/emb_{modality}_dim_{embedding_dim}_{emb_i}.pickle', 'wb') as w:
                    data = {dataset._id_to_name[idx]: emb[idx] for idx in dataset._id_to_name.keys()}
                    pickle.dump(data, w)


                # EVAL
                df = pd.read_csv(ROOT_DIR+'word_similarity/mturk-771.csv')
                vocab = pd.read_csv(ROOT_DIR+'intersect_vocab.csv')['name'].values
                indices = []

                for i in range(df.shape[0]):
                    if (df.loc[i,'word1'] in vocab) and (df.loc[i,'word2'] in vocab):
                        indices.append(i)

                df_sim = df.loc[indices,['word1','word2','similarity']].reset_index(drop=True)
                df_sim['similarity'] = df_sim['similarity']/df['similarity'].max()

                def calc_cosine_sim(a,b):
                    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

                def calc_eucl_sim(a,b):
                    return -np.linalg.norm(a-b)

                emb_sim = []

                for i in range(df_sim.shape[0]):
                    word1 = df_sim.loc[i,'word1']
                    word2 = df_sim.loc[i,'word2']
                    word1_emb = emb[dataset._name_to_id[word1],:]
                    word2_emb = emb[dataset._name_to_id[word2],:]
                    emb_sim.append(calc_cosine_sim(word1_emb, word2_emb))
                    # emb_sim.append(calc_eucl_sim(word1_emb, word2_emb))

                df_sim['emb_sim'] = emb_sim
                display(df_sim.corr())

