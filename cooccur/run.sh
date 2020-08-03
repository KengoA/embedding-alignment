# # Run download.sh first if no data directory is empty
# # Create intersection vocab
python ./vocab/make_vocab.py

# # Count cooccurrence for images
cd openimages
python ./src/aggregate_data.py
python ./src/count_cooccur.py

cd ..
cd enwiki
split --verbose -l 5000 data/raw/enwiki_201308.txt data/streams/enwiki_201308_
python src/preprocess.py
cd src/cython
python setup.py build_ext --inplace
python run.py
