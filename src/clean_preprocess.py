# installations
# !pip install nltk

# imports
import pandas as pd
import re
import nltk  # splitting data into sentences for NN
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.utils.class_weight import compute_class_weight  # used to balance classification model
import pickle

# read in data
df_transcript = pd.read_csv('data/cited_datasets/youtube_transcripts.csv')

# remove empty/null transcripts
df_transcript = df_transcript.dropna(subset=['transcript'])
df = df_transcript[df_transcript['transcript'].str.strip() != '']

# remove duplicates
df_transcript = df_transcript.drop_duplicates(subset=['transcript'])

# replace whitespaces
df_transcript['transcript'] = (
    df['transcript']
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

df_transcript['transcript_clean'] = df_transcript['transcript'].str.lower()

# simple cleaning of transcripts
def clean_transcript_lang(text):
    # remove bracket ([]) tags
    text = re.sub(r'\[[^\]]+\]', ' ', text)
    # remove headers like '>>'
    text = re.sub(r'>+\s*', ' ', text)
    # normalize
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_transcript['transcript_clean'] = df_transcript['transcript_clean'].apply(clean_transcript_lang)

# tokenize to check transcript lengths
df_transcript['token_count'] = df_transcript['transcript_clean'].str.split().str.len()

# filter out short transcripts
df_transcript = df_transcript[df_transcript['token_count'] >= 100]

# split lectures into 400 word sections to input for optimized training
def split_lectures(words, max_len=400):
    return ['' ''.join(words[i:i+max_len])
            for i in range(0, len(words), max_len)]

df_transcript['lecture_section'] = df_transcript['transcript_clean'].str.split().apply(split_lectures)

nltk.download('punkt')
nltk.download('punkt_tab')

def split_sentences(transcript):
    # first approach - typical tokenizing
    sentences = nltk.sent_tokenize(transcript)

    # add punctuation if needed
    if len(sentences) <= 1:
        # add breaks at commas, 'so', 'and then', etc.
        transcript = re.sub(r'\s*,\s*', '. ', transcript)
        transcript = re.sub(r'\s+so\s+', '. so ', transcript)
        transcript = re.sub(r'\s+and\s+', '. and ', transcript)

        # run tokenizer
        sentences = nltk.sent_tokenize(transcript)

    # remove extra whitespace
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]

df_transcript['sentence'] = df_transcript['transcript_clean'].apply(split_sentences)

# combine short sentences
def merge_short_sentences(sentences, min_len=40):
    merged = []
    for s in sentences:
        if not merged:
            merged.append(s)
        elif len(s) < min_len:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged

df_transcript['sentence'] = df_transcript['sentence'].apply(merge_short_sentences)

# create sentence dataframe
sentence_entries = []
for idx, row in df_transcript.iterrows():
    for i, sentence in enumerate(row['sentence']):
        sentence_entries.append({
            'transcript_id': idx,
            'sentence_id': i,
            'sentence': sentence,
            'title': row['title'],
            'playlist': row['playlist_name'],
        })

df_sentence = pd.DataFrame(sentence_entries)

# check number of transcripts per playlist
# df_transcript['playlist_name'].value_counts()

# see unique playlists
# df_transcript['playlist_name'].unique()

# group/cluster/put all topics into buckets
def group_topic(playlist_name: str) -> str:
    name_playlist = playlist_name.lower()

    # NLP/Transformers. Note w is the word(s) that is checked to group the data.
    if any(w in name_playlist for w in [
        'natural language processing', 'nlp', 'transformer', 'bert',
        'spaCy'.lower(), 'hugging face'
    ]):
        return 'NLP'

    # RL
    if 'reinforcement learning' in name_playlist or 'rl ' in name_playlist or 'rl -' in name_playlist:
        return 'RL'

    # ML / Deep Learning
    if any(w in name_playlist for w in [
        'machine learning', 'deep learning', 'neural network',
        'cs229', 'cs230', 'cs221', 'cs234', 'ml tech talks',
        'mit advanced machine learning', 'advanced deep learning',
        'deep learning course', 'deep learning basics',
        'practical deep learning', 'deep learning fundamentals',
        'intro to deep learning', 'deep learning from the foundations'
    ]):
        return 'ML/Deep Learning'

    # AI: General/Papers/Concepts
    if any(w in name_playlist for w in [
        'artificial intelligence', 'ai ', 'ai-', 'ai,', 'ai:',
        'age of a.i.', 'two minute papers', 'papers explained',
        'ai explained', 'cognitive and ai', 'deep learning research papers'
    ]):
        return 'AI: General/Papers'

    # DS/Stats
    if any(w in name_playlist for w in [
        'data science', 'pydata', 'scipy', 'tabular data',
        'practical statistics', 'data engineering', 'linear algebra',
        'data analysis', 'pandas tutorial', 'free data science course',
        'python data analysis'
    ]):
        return 'DS/Stats'

    # Programming/Python/Libraries
    if any(w in name_playlist for w in [
        'python', 'django', 'flask', 'pytorch', 'tensorflow',
        'matplotlib', 'modules and libraries', 'projects',
        'programming', 'web scraping', 'beautiful soup'
    ]):
        return 'Programming/Python'

    # Podcasts/Talks
    if any(w in name_playlist for w in [
        'podcast', 'deepmind: the podcast', 'lex fridman',
        'our story'
    ]):
        return 'Podcast/Talk'

    # Everything else
    return 'Other'

df_transcript['topic'] = df_transcript['playlist_name'].apply(group_topic)

# check number of transcripts per topic
# print(df_transcript['topic'].value_counts())

topics = df_transcript['topic']
classification_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(topics),
    y=topics
)

# just used for data visualization
# df_transcript.head()

# df_sentence.head()

# df_transcript

# df_sentence

df_transcript = df_transcript.reset_index().rename(columns={'index': 'transcript_id'})

# omit outliers
lecture_candidates = df_transcript[
    (df_transcript['token_count'] >= 300) &
    (df_transcript['token_count'] <= 2500)
]

# sampling three lectures per topic
def sample_lectures_per_topic(df):
  lecture_samples = []

  for topic, topic_bucket in df.groupby('topic'):
      lecture_samples_for_topic = topic_bucket.sample(n=3, random_state=42)
      lecture_samples.append(lecture_samples_for_topic)

  return pd.concat(lecture_samples).reset_index(drop=True)

lectures_to_hand_summarize = sample_lectures_per_topic(lecture_candidates)
lectures_to_hand_summarize['paraphrased_summary'] = ''
lectures_to_hand_summarize.to_csv('lectures_to_hand_summarize.csv', index=False)

# read in lectures that were hand summarized
df_summ = pd.read_csv('data/cleaned_datasets/lectures_hand_summarized.csv')

# create one final dataframe that allows sentences to be invididually evaluated per transcript
df_final = (
    df_sentence
    .merge(df_summ[['transcript_id', 'paraphrased_summary', 'topic']],
           on='transcript_id',
           how='inner')
)

# just added to visualize/verify the final dataframe
# print(df_final)

df_transcript.to_pickle('df_transcript.pkl')
df_summ.to_pickle('df_summ.pkl')
df_final.to_pickle('df_final.pkl')