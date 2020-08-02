import pandas as pd

if __name__=='__main__':
    df = pd.read_csv('class-descriptions-boxable.csv', header=None)
    df.columns = ['mid', 'name']
    df['name'] = df['name'].str.lower()

    print('original vocab size: {}'.format(df.shape[0]))

    for i in range(df.shape[0]):
        s = df.loc[i,'name']
        if '(' in s:
            df.loc[i,'name'] = s[:s.find('(')-1]

    idx = [i for i in range(df.shape[0]) if len(df.loc[i,'name'].split()) == 1]

    df = df.loc[idx,:]
    df.to_csv('intersect_vocab.csv', index=False)

    print('filtered vocab size: {}'.format(df.shape[0]))