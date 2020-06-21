import pandas as pd
import spacy
import streamlit as st
from spacy import displacy
import altair as alt

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
DEFAULT_TEXT = 'będę starał się rozwiązać kryzys wodny oraz zewaluować program 500+'

POLITICAL_LABELS = [
    'healhcare',
    'welfare',
    'society',
    'political_and_legal_system',
    'infrastructure_and_enviroment',
    'defense_and_security',
    'immigration',
    'education',
    'foreign_policy',
]


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("Interactive model visualizer")
st.sidebar.markdown(
    """
Process text with [spaCy](https://spacy.io) models and visualize named entities,
dependencies and more. Uses spaCy's built-in
[displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
"""
)
st.sidebar.title('Which model would you like to  use')
spacy_model = st.sidebar.selectbox(
    '',
    sorted(['pl_political_advertising_model']),
    index=0
)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
model_load_state.empty()

nlp = load_model(spacy_model)

st.header('Example of extracted spans')
text = st.text_area("Text to analyze", DEFAULT_TEXT)
doc = process_text(spacy_model, text)

st.header("Political Advertising")
st.sidebar.header("Categories")
label_set = [l for l in nlp.get_pipe("ner").labels if l in POLITICAL_LABELS]
labels = st.sidebar.multiselect(
    "", options=label_set, default=list(label_set)
)
html = displacy.render(doc, style="ent")
# Newlines seem to mess with the rendering
html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
data = [
    [str(getattr(ent, attr)) for attr in attrs]
    for ent in doc.ents
    if ent.label_ in labels
]
df = pd.DataFrame(data, columns=attrs)
st.dataframe(df)

st.header('Annotation Stats')

annotations_df = pd.read_pickle('datasets/political_advertising/pl_political_advertising_twitter_iter_1.pkl')
st.dataframe(annotations_df)

st.subheader('# of all annotations')
st.info(f'{len(annotations_df)}')
