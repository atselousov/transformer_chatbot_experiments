#!/bin/bash
mkdir checkpoints

mkdir datasets
cd datasets
wget -O ConvAI2.tar.gz https://s3.amazonaws.com/datasets.huggingface.co/ConvAI2.tar.gz
wget -O DailyDialog.tar.gz https://s3.amazonaws.com/datasets.huggingface.co/DailyDialog.tar.gz
tar -xvzf ConvAI2.tar.gz .
tar -xvzf DailyDialog.tar.gz .
rm ConvAI2.tar.gz
rm DailyDialog.tar.gz
cd ..

cd metrics/3rdparty/
wget -O meteor-1.5.tar.gz http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvzf ./meteor-1.5.tar.gz
rm meteor-1.5.tar.gz
cd ../..

wget -O parameters.tar.gz https://s3.amazonaws.com/models.huggingface.co/openai_gpt/parameters.tar.gz
tar -xvzf ./parameters.tar.gz
rm parameters.tar.gz

pip install -r requirements.txt
pip install torch
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon'); nltk.download('perluniprops'); nltk.download('punkt')"
python -m spacy download en

cd ..
git clone https://github.com/facebookresearch/ParlAI.git
cd ParlAI
python setup.py develop
