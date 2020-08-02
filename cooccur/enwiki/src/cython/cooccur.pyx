import json
from tqdm import tqdm

cdef increment_cooccur(str center, dict cooccur, list left_window, list right_window):
    cdef str context_word
    cdef float dist
    cdef char left_window_len = len(left_window)
    cdef char right_window_len = len(right_window)

    for context_word in cooccur[center].keys():
        if context_word in left_window:
            dist = left_window_len - left_window.index(context_word) 
            cooccur[center][context_word] += 1/dist
        
        if context_word in right_window:
            dist = right_window.index(context_word) + 1 
            cooccur[center][context_word] += 1/dist


cpdef count_cooccur(str path, list vocab):
    cdef char WINDOW_SIZE = 15
    cdef list corpus, left_window, right_window 
    cdef dict cooccur
    cdef str center, concept, context
    cdef int i, corpus_size

    with open(path, 'r') as r:
        cooccur = {concept: {context: 0 for context in vocab if context != concept} for concept in vocab}
        corpus = r.readline().split()
        corpus_size = len(corpus)
        for i in tqdm(range(corpus_size)):
            if corpus[i] in vocab:
                center = corpus[i]
                if i < WINDOW_SIZE:
                    left_window = corpus[:i]
                    right_window = corpus[i+1:i+WINDOW_SIZE+1]
                elif i > corpus_size-WINDOW_SIZE:
                    left_window = corpus[i-WINDOW_SIZE:i]
                    right_window = corpus[i:]
                else:
                    left_window = corpus[i-WINDOW_SIZE:i]
                    right_window = corpus[i+1:i+WINDOW_SIZE+1]
                
                increment_cooccur(center, cooccur, left_window, right_window)
                del center
                del left_window
                del right_window
        

        with open('../data/intermediate/cooccur_{}'.format(path.split('_')[2][:-4]), 'w') as w:
            json.dump(cooccur, w)
    
    del corpus
    del cooccur
    del vocab