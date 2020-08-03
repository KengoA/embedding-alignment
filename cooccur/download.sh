# download class names
wget -O ./vocab/class_names.csv https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv

# download Open-Images dataset for image labels
wget -O ./open-images/data/raw/train.csv https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv
wget -O ./open-images/data/raw/validation.csv https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv
wget -O ./open-images/data/raw/test.csv https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv

# # download and extract wikipedia dump from 2013 August
# wget 
# bzip2 -d 