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

## Input Format

The files used for training, validation and testing must be in a format similar to the [CoNLL](https://universaldependencies.org/format.html): 

```
# sent_id = email-enronsent20_01-0048
# text = Please let us know if you have additional questions.
1	Please	please	INTJ	UH	_	2	discourse	2:discourse	_
2	let	let	VERB	VB	Mood=Imp|VerbForm=Fin	0	root	0:root	_
3	us	we	PRON	PRP	Case=Acc|Number=Plur|Person=1|PronType=Prs	2	obj	2:obj|4:nsubj:xsubj	_
4	know	know	VERB	VB	VerbForm=Inf	2	xcomp	2:xcomp	_
5	if	if	SCONJ	IN	_	7	mark	7:mark	_
6	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	7	nsubj	7:nsubj	_
7	have	have	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	4	advcl	4:advcl:if	_
8	additional	additional	ADJ	JJ	Degree=Pos	9	amod	9:amod	_
9	questions	question	NOUN	NNS	Number=Plur	7	obj	7:obj	SpaceAfter=No
10	.	.	PUNCT	.	_	2	punct	2:punct	_
```

## Training

To train a model, use the `train.py` script. This will start training a model that will predict the labels of the column specified by the `[predict_column]` argument.

```
python3 train.py [path_train_file] [path_dev_file] [tokens_column] [predict_column] [lang_model_name]
```

## Inference

To predict new values, use the `predict.py` script. This will create a new file by replacing the predicted column of the test file with the predicted values.

```
python3 predict.py [path_test_file] [model_path] [tokens_column] [predict_column] [lang_model_name]
```

## Results

#### English EWT

| model | upos | xpos | 
| --- | --- | --- |
| bert-base-cased | - | - |
| roberta-base | - | - |
| distilbert-base-cased | - | - |



