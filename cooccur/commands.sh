source ./venv/bin/activate
python vocab/make_vocab.py
cd enwiki
split --verbose -l 5000 data/raw/enwiki_201308.txt data/streams/enwiki_201308_
python preprocess.py
cd cython
python setup.py build_ext --inplace
python run.py
