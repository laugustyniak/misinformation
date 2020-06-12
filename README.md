# Misinformation Analysis in Social Media

Our group of researchers works on misinformation analysis in Social Media. The area of misinformation analysis is not so widely known. Moreover, there also do not exist any Machine Learning-based solutions for working with Polish text in this area. We want to fill this gap. We create and openly publish datasets and well as trained models that can be used to analyzed Polish Social Media content on a vast scale. 

# Political Advertising Dataset and Tagging Model

Political campaigns are full of political ads posted by candidates on social media. Political advertisement constitute a basic form of campaigning, subjected to various social requirements. 

We present the first publicly open dataset for detecting specific text chunks and categories of political advertising in the Polish language. It contains 1,705 human-annotated tweets tagged with nine categories, which constitute campaigning under Polish electoral law.

We achieved a 0.65 inter-annotator agreement (Cohen's kappa score). An additional annotator resolved the mismatches between the first two annotators improving the consistency and complexity of the annotation process. \

## Model loading and usage

We trained a Convolutional Neural Network model using a spaCy Named Entity classifier, achieving a 70\% F1 score for 5-fold cross-validation. We used fastText vectors for Polish and default spaCy model hyperparameters.

```python
import spacy
nlp = spacy.load('')

```

## Dataset loading

```python
import pandas as pd
df = pd.read_pickle('datasets\political_advertising\pl_political_advertising_twitter_iter_1.pkl')
```

## ATTENTION: Not raw text of tweets available 

The pickled file do not contain text of tweets due to Twitter policy. It contains spans annotations, tweet id, url for a tweet and sentiment orientations assigned to the tweet. 

If you want to get tweet text you should either download them by your self via Twitter API or contact with us and will share the whole dataset to our collaborators.  

![DATASET_DF](/imgs/dataset_df.png) 


 

# References 



```bibtex

```
