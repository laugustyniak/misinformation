# Misinformation Analysis in Social Media

Our group of researchers works on misinformation analysis in Social Media. The area of misinformation analysis is not so widely known. Moreover, there also do not exist any Machine Learning-based solutions for working with Polish text in this area. We want to fill this gap. We create and openly publish datasets and well as trained models that can be used to analyzed Polish Social Media content on a vast scale. 

# Political Advertising Dataset and Tagging Model

Political campaigns are full of political ads posted by candidates on social media. Political advertisement constitute a basic form of campaigning, subjected to various social requirements. 

We present the first publicly open dataset for detecting specific text chunks and categories of political advertising in the Polish language. It contains 1,705 human-annotated tweets tagged with nine categories, which constitute campaigning under Polish electoral law.

We achieved a 0.65 inter-annotator agreement (Cohen's kappa score). An additional annotator resolved the mismatches between the first two annotators improving the consistency and complexity of the annotation process.

## Interactive dashboards

Annotations Statistics TODO

Model prediction capabilities and use case TODO 

## Model loading and usage

We trained a Convolutional Neural Network model using a spaCy Named Entity classifier, achieving a 70\% F1 score for 5-fold cross-validation. We used fastText vectors for Polish and default spaCy model hyperparameters.

[Download model here](https://drive.google.com/file/d/1Lq9I6NmDG3VV-vp7WrYx3HCEdToeEymG/view?usp=sharing) 

### Usage via conda 

```
conda create --name political_advertising  python=3.8 -y
conda activate political_advertising
```

Install spaCy model with political advertising categories via 
```bash
pip install -r requirements.txt
pip install /path/to/pl_political_advertising_model-1.0.0.tar.gz
```

Then open python or jupyter notebook and run

```python
import spacy
nlp = spacy.load('pl_political_advertising_model')
[[ent.label_ for ent in doc.ents] for doc in nlp.pipe(['walczmy o niezależność sądów', 'będę starał rozwiązać kryzys wodny, który nastąpi w ciągu nabliższych X lat'])]
# [['political_and_legal_system'], ['infrastructure_and_enviroment']]
```

## Dataset loading

```python
import pandas as pd
df = pd.read_pickle('datasets\political_advertising\pl_political_advertising_twitter_iter_1.pkl')
```

## ATTENTION: Not raw text of tweets available 

The pickled file does not contain the text of tweets due to the [Twitter policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy). It contains spans annotations, tweet id, URL for a tweet, and sentiment orientations assigned to the tweet.
![DATASET_DF](/imgs/dataset_df.png)
 
**If you want to get tweet text**, you should either download them yourself via Twitter API (using tweet ids or their URLs) or **contact us, and we will share the whole dataset with our collaborators (much more relaxed and faster solution).**  
 
# Citing


```bibtex
@misc{augustyniak2020political,
    title={Political Advertising Dataset: the use case of the Polish 2020 Presidential Elections},
    author={Łukasz Augustyniak and Krzysztof Rajda and Tomasz Kajdanowicz and Michał Bernaczyk},
    year={2020},
    eprint={2006.10207},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
