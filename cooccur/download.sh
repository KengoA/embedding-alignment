### Image: Open Images ###
# download class names
wget -O ./vocab/class_names.csv https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv

# download Open-Images dataset for image labels
wget -O ./open-images/data/raw/train.csv https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv
wget -O ./open-images/data/raw/validation.csv https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv
wget -O ./open-images/data/raw/test.csv https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv


### Text: Wikipedia Dump ###
# download and extract wikipedia dump from 2013 August
export fileid=1uBbwHMCPvX9QeOoEfl68dy7jWj4iF87E
export filename=enwiki/data/raw/enwiki_201308.txt.bz2

## WGET ##
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

rm confirm.txt cookies.txt

cd enwiki/data/raw/
bzip2 -d -v -k enwiki_201308.txt.bz2