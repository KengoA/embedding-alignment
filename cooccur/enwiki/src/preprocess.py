import glob
import string
import time
import numpy as np
from joblib import Parallel, delayed

def append_tokens_from_txt(write_dir, read_path):
    punct = '!"#$%&\'()*+,ー—＋./…:;<=>?@[\\]^_`{|}~“”‘’、。＋－＝＜＞「」｛｝'
    count = 0
    with open(read_path, 'r') as r:
        with open(write_dir+'enwiki_{}.txt'.format('_'.join(read_path.split('_')[-2:])), "w") as w:
            start_t = time.time()
            tokens = []
            for line in r.readlines():
                # skip lines with article headers and subheaders
                if line[0] != '#':
                    # lowercasing and punctuation removal
                    l = line.lower().translate(str.maketrans('', '', punct)).split()
                    for token in l:
                        # token length larger than 2 and not starting with a number
                        if (len(token) >= 2) and (token[0] not in string.digits):
                            tokens.append(token)
                            count += 1
            w.writelines(' '.join(tokens))
            del tokens
            
            stream_time = round(time.time()-start_t,2)
            stream_times.append(stream_time)

            print("== Time: {0:.2f} secs == Tokens: {1:.2f} M ==".format(
                stream_time,
                round(count/1000000,2))
                )

if __name__ == "__main__":
    write_dir = './data/preprocessed/'
    read_dir = './data/streams/'

    read_paths = sorted(glob.glob(read_dir+'enwiki_*'))
    stream_times = []
    print("Preprocessing {} files under {} ...".format(len(read_paths), read_dir))
    Parallel(n_jobs=-1, verbose=10)(delayed(append_tokens_from_txt)(write_dir, read_path) for read_path in read_paths)