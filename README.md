# BERT-Sequence-Labeling

This repostiory integrates [HuggingFaces](https://github.com/huggingface)'s models in an end-to-end pipeline for sequence labeling. [Here](https://huggingface.co/transformers/pretrained_models.html) 
is a complete list of the available models. 

If you found this repository helpful, please give it a star.:blush:

## Install

```
git clone https://github.com/avramandrei/BERT-Sequence-Labeling.git
cd BERT-Sequence-Labeling
pip3 install -r requirements.txt
```

## Train a Model

The files used for training must be in a format similar to the [CoNLL](https://universaldependencies.org/format.html). 

For instance, to train a POS tagger on [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html), use the following 
command:

```
python3 train.py [path_train_ewt] [path_dev_ewt] 3 --token_column 1
```

This will automatically start training a `bert-base-cased` model. To change the model use the `--lang_model_name` argument.

## Inference

Use the `predict.py` script to infer new values. For instance, predict the POS of the [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html) test file:

```
python3 predict.py [path_test_ewt] [model_path] 3
```
