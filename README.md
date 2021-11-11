# Wrap ups

## Preparation

### Creating an environment:

    conda create -n textseg python=2.7 numpy scipy ipython 
    source activate textseg
    pip install http://download.pytorch.org/whl/cu80/torch-0.3.0-cp27-cp27mu-linux_x86_64.whl 
    pip install gensim
    pip install nltk==3.0
    pip install tqdm pathlib2 segeval tensorboard_logger flask flask_wtf 
    pip install pandas xlrd xlsxwriter termcolor

### Prepare data
- Download `word2vec` model: 
   - BaiduNetDisk: https://pan.baidu.com/s/1ESjl4zG8M9-Bfx30bNcmVA 
      - passsword: 9p6r
   - [One Drive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xwangeg_connect_ust_hk/EZ-FG2a4L-9NvjkqATec__8Bn1leLla3rA9guj56bAcSgw?e=Dlz2uf)

- After unzipping the file, you will get two files: `word2vec.bin` and `word2vec.bin.vectors.npy`. Put them in the `data/word2vec/`


## Run segmentation model on user-defined texts
    python run_seg.py --model <path_to_model> --word2vec <path_to_word2vec> --str <target_texts_to_be_segmented>   

## Others
### Speed up the word2vec loading process for orginal word2vec
- https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time


----
# Text Segmentation as a Supervised Learning Task

This repository contains code and supplementary materials which are required to train and evaluate a model as described in the paper [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337)

## Downalod required resources

wiki-727K, wiki-50 datasets:
>  https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0

word2vec:
>  https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM



Fill relevant paths in configgenerator.py, and execute the script (git repository includes Choi dataset)

## Creating an environment:

    conda create -n textseg python=2.7 numpy scipy gensim ipython 
    source activate textseg
    pip install http://download.pytorch.org/whl/cu80/torch-0.3.0-cp27-cp27mu-linux_x86_64.whl 
    pip install tqdm pathlib2 segeval tensorboard_logger flask flask_wtf nltk
    pip install pandas xlrd xlsxwriter termcolor

## How to run training process?

    python run.py --help

Example:

    python run.py --cuda --model max_sentence_embedding --wiki 

## How to evaluate trained model (on wiki-727/choi dataset)?

    python test_accuracy.py  --help

Example:

    python test_accuracy.py --cuda --model <path_to_model> --wiki



## How to create a new wikipedia dataset:
    python wiki_processor.py --input <input> --temp <temp_files_folder> --output <output_folder> --train <ratio> --test <ratio>

Input is the full path to the wikipedia dump, temp is the path to the temporary files folder, and output is the path to the newly generated wikipedia dataset.

Wikipedia dump can be downloaded from following url:

> https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2



