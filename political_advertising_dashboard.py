import altair as alt
import pandas as pd
import seaborn as sns
import spacy
import streamlit as st
from spacy import displacy

from political_advertising.categories import POLITICAL_ADVERTISING_CATEGORIES_EN_TO_PL, POLITICAL_LABELS

sns.set(style="ticks", color_codes=True)

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; 
border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
DEFAULT_TEXT = 'będę starał się rozwiązać kryzys wodny oraz zewaluować program 500+'

ANDRZEJ = 'Andrzej Duda'
RAFAL = 'Rafał Trzaskowski'


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


# st.sidebar.title('Which model would you like to  use?')
st.sidebar.title('Który model chcesz użyć?')
spacy_model = st.sidebar.selectbox(
    '',
    sorted(['pl_political_advertising_model']),
    index=0
)
model_load_state = st.info(f"Model ładuje się '{spacy_model}'...")
model_load_state.empty()

nlp = load_model(spacy_model)

st.header('Teskt poddany analizie')
text = st.text_area("Możesz wpisać poniżej własny tekst", DEFAULT_TEXT)
doc = process_text(spacy_model, text)

st.sidebar.header("Kategorie obietnic wyborczych")
label_set = [
    POLITICAL_ADVERTISING_CATEGORIES_EN_TO_PL[l.replace('_', ' ')]
    for l
    in nlp.get_pipe("ner").labels
    if l in POLITICAL_LABELS
]
labels = st.sidebar.multiselect(
    "", options=label_set, default=list(label_set)
)

st.sidebar.header('Authors')
st.sidebar.info("""
[Łukasz Augustyniak](https://www.linkedin.com/in/lukaszaugustyniak/)

[Krzysztof Rajda](https://www.linkedin.com/in/krzysztof-rajda/)

[Tomasz Kajdanowicz](https://www.linkedin.com/in/kajdanowicz/)

[Michał Bernaczyk](https://www.linkedin.com/in/michal-bernaczyk-9741a22/)
""")

st.header("Wykryte kategorie obietnic wyborczych")
html = displacy.render(doc, style="ent")
# Newlines seem to mess with the rendering
html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

st.header('Kampania Prezydencka 2020')
st.subheader('Analiza wybranych kont Kandydatów na Prezydenta RP')
st.markdown("""
<iframe src=https://twitframe.com/show?url=https://twitter.com/AndrzejDuda2020/status/1280464004385832962 width=49%></iframe>
<iframe src=https://twitframe.com/show?url=https://twitter.com/AndrzejDuda/status/1281135770481446912 width=49%></iframe>
<iframe src=https://twitframe.com/show?url=https://twitter.com/trzaskowski2020/status/1281152965794050049 width=49%></iframe>
<iframe src=https://twitframe.com/show?url=https://twitter.com/trzaskowski_/status/1280115100054151175 width=49%></iframe>
""", unsafe_allow_html=True)

candidates_tweets_df = pd.read_pickle('datasets/twitter_presidential_elections/komitety_processed.pkl')
# twitter_df = pd.read_pickle('datasets/twitter_presidential_elections/twitter_processed.pkl')

st.info(f'Zakres czasowy analizy: {candidates_tweets_df.full_date.min()} - {candidates_tweets_df.full_date.max()}')

candidate_tweet_count_df = pd.DataFrame(candidates_tweets_df.screen_name.value_counts()).reset_index()
candidate_tweet_count_df.columns = ['Kandydat', 'Sumaryczna liczba tweetów']

st.write(alt.Chart(candidate_tweet_count_df).mark_bar(
    color='red',
    opacity=0.3,
).encode(
    y='Kandydat',
    x='Sumaryczna liczba tweetów',
))

candidate_tweet_count_df = pd.DataFrame(candidates_tweets_df.name.value_counts()).reset_index()
candidate_tweet_count_df.columns = ['Kandydat', 'Sumaryczna liczba tweetów']

st.write(alt.Chart(candidate_tweet_count_df).mark_bar(
    color='red',
    opacity=0.3,
).encode(
    y='Kandydat',
    x='Sumaryczna liczba tweetów',
))

st.header('Liczba tweetów na przestrzeni obu kampanii prezydenckich')
candidate_tweets_count = {
    candidate: candidate_df.groupby(candidate_df.date).count()
    for candidate, candidate_df in candidates_tweets_df.groupby('name')
}
st.info(f"Średnia liczba tweetów dziennie {ANDRZEJ}")
st.warning(f"{candidate_tweets_count[ANDRZEJ]['id'].mean():.0f}")
st.info(f"Średnia liczba tweetów dziennie {RAFAL}")
st.warning(f"{candidate_tweets_count[RAFAL]['id'].mean():.0f}")

st.subheader('Liczba tweetów na przestrzeni ostatnich miesięcy')
df_a = pd.DataFrame(candidate_tweets_count[ANDRZEJ]['id']).reset_index()
df_a.columns = ['Data', 'Liczba tweetów']
df_a['Kandydat'] = ANDRZEJ

df_r = pd.DataFrame(candidate_tweets_count[RAFAL]['id']).reset_index()
df_r.columns = ['Data', 'Liczba tweetów']
df_r['Kandydat'] = RAFAL

df = df_a.append(df_r)

st.write(
    alt.Chart(df).mark_bar().encode(
        x='Data:O',
        y='Liczba tweetów:Q',
        color='Kandydat:N',
        row='Kandydat:N'
    )
)

st.header('Średnie nastawienie emocjonalne tweetów napisanych przez kandydatów')
st.info('gdzie -1 oznacza negatywne nastawienie, 1 pozytywne, 0 to neutralne.')
candidate_sentiment = {
    candidate: candidate_df.sentiment.mean()
    for candidate, candidate_df
    in candidates_tweets_df.groupby('name')
}
st.write(candidate_sentiment)

st.header('Średnie nastwienie emocjonalne tweetów w kontekście obietnic wyborczych')


def draw_political_advertising_categories_and_sentiment(
        df: pd.DataFrame,
        political_advertising_categories_en_to_pl_mapping,
        x_label
):
    df['Kategoria obietnicy wyborczej'] = df.category.apply(
        lambda c: political_advertising_categories_en_to_pl_mapping[c])
    df[x_label] = df['sentiment']
    df['Kandydat'] = df['candidate']

    st.write(
        alt.Chart(df).mark_bar().encode(
            x='Kategoria obietnicy wyborczej:O',
            y=f'{x_label}:Q',
            color='Kandydat:N',
            column='Kandydat:N'
        )
    )
    # g = sns.catplot(
    #     x="Kategoria obietnicy wyborczej",
    #     y=x_label,
    #     hue="Kandydat",
    #     kind="bar",
    #     data=df,
    # )
    # g.set_xticklabels(rotation=90)
    # st.pyplot()


candidate_categories_with_sentiment_df = pd.DataFrame([
    (row['name'], political_ad, row.sentiment)
    for _, row in candidates_tweets_df.iterrows()
    for political_ad in row.political_advertising_labels
], columns=['candidate', 'category', 'sentiment'])

draw_political_advertising_categories_and_sentiment(
    candidate_categories_with_sentiment_df.groupby(['candidate', 'category']).agg('mean').reset_index(),
    POLITICAL_ADVERTISING_CATEGORIES_EN_TO_PL,
    'Średnie nastawienie emocjonalne wpisów'
)

draw_political_advertising_categories_and_sentiment(
    candidate_categories_with_sentiment_df.groupby(['candidate', 'category']).agg('count').reset_index(),
    POLITICAL_ADVERTISING_CATEGORIES_EN_TO_PL,
    'Liczba obietnic wyborczych'
)

st.warning('Poniżej analiza tweetów napisanych przez wyborców między 2020-06-28, a 2020-07-07')
st.warning('W sumie zbiór zawiera 162 397 tweetów')
st.warning('Pobranych korzystając z następujących hastagów: #Trzaskowski2020, #Duda2020, #Wybory2020, #WyboryPrezydenckie2020')

st.header(
    'Średnie nastawienie emocjonalne tweetów napisanych przez wybórców w kontekście kategorii obietnic wyborczych'
    ' bez neutralnych tweetów')

twitter_sentiment_df = pd.read_csv('datasets/twitter_presidential_elections/twitter_categories_with_sentiment.csv')
twitter_sentiment_divided_df = pd.read_csv(
    'datasets/twitter_presidential_elections/twitter_categories_with_sentiment_divided.csv')
st.write(
    alt.Chart(twitter_sentiment_df).mark_bar().encode(
        x='Kategoria obietnicy wyborczej:O',
        y=f'Średnie nastawienie emocjonalne wpisów:Q',
    ).properties(
        width=400,
        height=550
    )
)

twitter_sentiment_divided_df['Suma sentymentu (wystąpienia)'] = twitter_sentiment_divided_df['Liczba wystąpień'] * twitter_sentiment_divided_df['Sentyment']
st.write(
    alt.Chart(twitter_sentiment_divided_df).mark_bar().encode(
        x='Kategoria obietnicy wyborczej:O',
        y=f'Suma sentymentu (wystąpienia):Q',
        color='Sentyment:N',
        column='Sentyment:N'
    )
)

st.header('Liczba wystąpień obietnic wyborczych dla każdej kategorii')
st.write(
    alt.Chart(twitter_sentiment_df).mark_bar().encode(
        x='Kategoria obietnicy wyborczej:O',
        y=f'Liczba wystąpień obietnicy wyborczej:Q',
    ).properties(
        width=400,
        height=550
    )
)

# GOOGLE ANALYTICS
st.markdown("""
<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-TXZW3XX');</script>
<!-- End Google Tag Manager -->

<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-TXZW3XX"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
""", unsafe_allow_html=True)
