import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import argparse
import random

# Integrate tqdm with pandas
tqdm.pandas()

# GPU check
device = 0 if torch.cuda.is_available() else -1
print(f"Device used: {'GPU' if device == 0 else 'CPU'}")

# Turkish -> English topic mapping
label_map = {
    "göç": "migration",
    "ekonomi": "economy",
    "eğitim": "education",
    "sağlık": "health",
    "adalet": "justice",
    "dış politika": "foreign_policy",
    "enerji": "energy",
    "yerel yönetim": "local_government",
    "taziye": "condolence",
    "tebrik": "congratulation",
    "kültür": "culture",
    "spor": "sports",
    "afet": "disaster",
    "genel": "general"
    }

# Default (for MODEL) Turkish topic labels
default_candidate_labels = list(label_map.keys())

# Parse command line arguments
parser = argparse.ArgumentParser(description='Tweet topic classification')
parser.add_argument('--topics', nargs='*', help='User-defined topics (Turkish or English, comma-separated)')
args = parser.parse_args()

# Process user-defined topics if provided (accepts Turkish or English)
if args.topics:
    raw = [t.strip().lower() for t in ' '.join(args.topics).split(',')]
    reverse_map = {v: k for k, v in label_map.items()}
    candidate_labels = []
    for t in raw:
        if t in label_map:               # already Turkish
            candidate_labels.append(t)
        elif t in reverse_map:           # English provided -> convert to Turkish
            candidate_labels.append(reverse_map[t])
        else:
            print(f"Warning: '{t}' not recognized and skipped.")
    if not candidate_labels:
        print("No valid topic found, using defaults.")
        candidate_labels = default_candidate_labels
    print(f"Topics to be used in model (Turkish): {candidate_labels}")
else:
    candidate_labels = default_candidate_labels
    print("Default Turkish topics are used.")

print(f"Total number of topics (model): {len(candidate_labels)}")

# Prepare zero-shot classifier
classifier = pipeline("zero-shot-classification", 
                        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 
                        device=device,multi_label=True)

# Load tweet dataframe
df = pd.read_csv("data_processed/all_cleaned_tweets.csv")

# Batch classification function (Turkish label -> save as English)
def classify_topics_batch(texts_batch):
    results = classifier(texts_batch, candidate_labels=candidate_labels)
    topics = []
    for result in results:
        max_score = result["scores"][0]
        top_label_tr = result["labels"][0]
        if max_score < 0.5:
            top_label_tr = "belirsiz"
        topic_en = label_map.get(top_label_tr, "uncertain")
        topics.append(topic_en)
    return topics

# Batch size
batch_size = 8
topics = []

# Processing loop
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["Text"][i:i+batch_size].tolist()
    batch_topics = classify_topics_batch(batch_texts)
    topics.extend(batch_topics)

# Results are saved with English labels
df["topic"] = topics
df.to_csv("data_processed/all_cleaned_tweets_with_topics.csv", index=False)

# Topic distribution (English)
print("\n=== TOPIC DISTRIBUTION (English) ===")
topic_counts = df["topic"].value_counts()
for topic, count in topic_counts.items():
    print(f"{topic}: {count} tweets")

# Print 5 random samples from each topic
print("\n=== 5 RANDOM SAMPLES FROM EACH TOPIC ===")
for topic in topic_counts.index:
    topic_tweets = df[df["topic"] == topic]["Text"].tolist()
    if len(topic_tweets) > 0:
        sample_size = min(5, len(topic_tweets))
        random_samples = random.sample(topic_tweets, sample_size)
        
        print(f"\n--- {topic.upper()} ({sample_size} samples) ---")
        for i, tweet in enumerate(random_samples, 1):
            print(f"{i}. {tweet[:200]}{'...' if len(tweet) > 200 else ''}")