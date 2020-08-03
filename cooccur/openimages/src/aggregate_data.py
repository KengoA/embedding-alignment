import glob
import pandas as pd

if __name__ == "__main__":
    print('concatenating open-images dataset ...')
    df = pd.concat([pd.read_csv(path) for path in glob.glob('./data/raw/*.csv')])
    df = df[df['Confidence'] >= 0.5]
    df.to_csv('./data/preprocessed/open-images.csv', index=None)
    print('done.')