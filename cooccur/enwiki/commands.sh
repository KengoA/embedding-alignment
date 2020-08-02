split --verbose -l 100000 data/raw/enwiki_201308.txt data/streams/enwiki_201308_
python3 preprocess.py

cd cython
python3 setup.py build_ext --inplace
python3 test.py
python3 run.py