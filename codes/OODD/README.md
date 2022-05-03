## Structure of Data
DataProcessed
	dataset_name
		train
			Period1
				class1.train
				class2.train
				class3.train
                workspace_list

		dev
			Period1_class4.train
			Period1_class4.test
			Period1_class5.train
			Period1_class5.test
            workspace_list

		test
			Period1_class4.train
			Period1_class4.test
			Period1_class5.train
			Period1_class5.test
            workspace_list

Keep dev and test folders same.

## Dataset Format
Each line in the data file is an example instance, such that the input text is followed by the class label, separated by '\t'
Example - I need my card soon.	card_delivery_estimate

## Word Embeddings
In config/config
	w2vfile=../glove/glove.6B.100d.txt for EN
			../glove/Spanish_BERT_Multilingual.txt     768 Dimension
			../glove/Thai_BERT_Multilingual.txt        768 Dimension

Make embeddings using glove/BERTMaker.ipynb
If you do not want to make it, in config/config give w2vfile as 'BERT_Multilingual'. But this process will be slower as it gets embeddings one word at a time.

## Run Code
Edit run.sh
PERIOD=1
DATASET = dataset_name
GPU=as required

OOD Class Label = UNCONFIDENT_INTENT_FROM_SLAD

[Original GitHub Repository](https://github.com/SLAD-ml/few-shot-ood)