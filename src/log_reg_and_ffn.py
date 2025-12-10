# imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ffn_summ import *
import pickle

# read in final dataframe from cleaning/preprocessing
df_final = pd.read_pickle('src/pkl_files/df_final.pkl')

# max number of sentences for a generated summary - "average" across handwritten summaries
SUMMARY_MAX_LEN = 10

# using tfidf to vectorize sentence importance
tfidf = TfidfVectorizer(max_features=10000,
                        ngram_range=(1, 2),
                        stop_words='english')
tfidf.fit(
    pd.concat([df_final['sentence'],
               df_final['paraphrased_summary']], axis=0)
)

# build sentence importance labels
def build_sentence_importance_labels(df):
    labels = []

    for tid, group in df.groupby('transcript_id'):
        sentence_texts = group['sentence'].tolist()
        summary_text = group['paraphrased_summary'].iloc[0]

        sentence_vecs = tfidf.transform(sentence_texts)
        summary_vec  = tfidf.transform([summary_text])

        similarity_scores = cosine_similarity(sentence_vecs, summary_vec).flatten()

        # Top K indices (summary-worthy)
        k = min(SUMMARY_MAX_LEN, len(similarity_scores))
        sentence_idxs = np.argsort(similarity_scores)[-k:]

        y = np.zeros_like(similarity_scores, dtype=int)
        y[sentence_idxs] = 1

        labels.extend(y.tolist())

    df['label'] = labels
    return df

df_final = build_sentence_importance_labels(df_final)

# once again added to visualize/verify the final dataframe
# print(df_final)

# checking that there's enough of both. this is neat - this means for about every five sentences, one is actually important.
# later on realized this would be a challenge...
# df_final['label'].value_counts()

# split into train/val/test sets for training
y = df_final['label'].to_numpy()
X_tfidf = tfidf.transform(df_final['sentence'])

ids_all = df_final['transcript_id'].unique()
ids_train, ids_rest = train_test_split(ids_all, test_size=0.2, random_state=42)
ids_val, ids_test = train_test_split(ids_rest, test_size=0.50, random_state=42)

mask_train = df_final['transcript_id'].isin(ids_train).to_numpy()
mask_test  = df_final['transcript_id'].isin(ids_test).to_numpy()
mask_val   = df_final['transcript_id'].isin(ids_val).to_numpy()

X_tfidf_train = X_tfidf[mask_train]
X_tfidf_test  = X_tfidf[mask_test]
X_tfidf_val  = X_tfidf[mask_val]
y_train = y[mask_train]
y_test  = y[mask_test]
y_val = y[mask_val]

# training logistic regression model
log_reg_model = LogisticRegression(
    max_iter=500,
    class_weight='balanced',
    solver='liblinear'
)
log_reg_model.fit(X_tfidf_train, y_train)

# evaluate logistic regression model
y_pred_val = log_reg_model.predict(X_tfidf_val)
print(classification_report(y_val, y_pred_val))

y_pred_test = log_reg_model.predict(X_tfidf_test)
print(classification_report(y_test, y_pred_test))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_tensor_train = torch.tensor(X_tfidf_train.toarray(), dtype=torch.float32).to(device)
X_tensor_test  = torch.tensor(X_tfidf_test.toarray(),  dtype=torch.float32).to(device)
X_tensor_val   = torch.tensor(X_tfidf_val.toarray(),   dtype=torch.float32).to(device)

y_tensor_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_tensor_test  = torch.tensor(y_test,  dtype=torch.float32).to(device)
y_tensor_val   = torch.tensor(y_val,   dtype=torch.float32).to(device)

input_dim = X_tfidf_train.shape[1]
feedforward_net_summary = FF_Net_Summary(input_dim).to(device)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
pos_weight_value = n_neg / n_pos

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer_ffn_summary = optim.Adam(feedforward_net_summary.parameters(),  lr=0.001, weight_decay=1e-4)

batch_size = 64
num_epochs = 25

# training feedforward neural network model
for epoch in range(num_epochs):
    feedforward_net_summary.train()
    permutation = torch.randperm(X_tensor_train.size(0))

    epoch_loss = 0

    for i in range(0, X_tensor_train.size(0), batch_size):
        idxs = permutation[i:i+batch_size]
        X_selected = X_tensor_train[idxs]
        y_selected = y_tensor_train[idxs]

        optimizer_ffn_summary.zero_grad()

        outputs = feedforward_net_summary(X_selected).squeeze(1)
        loss = criterion(outputs, y_selected)
        loss.backward()
        optimizer_ffn_summary.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f}')

# evaluate accuracy of a model
def evaluate(model, X_tensor, y_true, threshold=0.35):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= threshold).astype(int)

    print(classification_report(y_true, preds))
    return probs, preds

# evaluate accuracy of FFN
print('validation performance')
probs_val, preds_val = evaluate(feedforward_net_summary, X_tensor_val, y_val)

print('test performance')
probs_test, preds_test = evaluate(feedforward_net_summary, X_tensor_test, y_test)

# save tfidf vectorizer
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# save logistic regression model
with open('log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg_model, f)

# save test set
np.save('ids_test.npy', ids_test)

# save input_dim of FFN
np.save('input_dim.npy', np.array([input_dim]))

# save FFN model
torch.save(feedforward_net_summary.state_dict(), 'ffn_summary.pth')
