import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Set Turkish locale for plots
plt.rcParams['font.family'] = ['DejaVu Sans']

# Load emotion definitions
emotions_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/emotions_english_turkish.csv')
emotion_name_to_id = dict(zip(emotions_df['emotion_name_en'], emotions_df['emotion_id']))
emotion_id_to_name = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_en']))
emotion_id_to_name_tr = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_tr']))

# Load cleaned tweets dataset
print("Loading cleaned tweets dataset...")
cleaned_tweets_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/3_tweets_with_emotions/cleaned_tweets_with_emotions.csv')
print(f"Loaded {len(cleaned_tweets_df)} tweets")
print(f"Columns: {list(cleaned_tweets_df.columns)}")

# Basic dataset information
print(f"\n=== DATASET OVERVIEW ===")
print(f"Total tweets: {len(cleaned_tweets_df)}")
print(f"Date range: {cleaned_tweets_df['Date'].min()} to {cleaned_tweets_df['Date'].max()}")
print(f"Unique authors: {cleaned_tweets_df['Author'].nunique()}")
print(f"Parties: {cleaned_tweets_df['party'].unique()}")
print(f"Political sides: {cleaned_tweets_df['political_side'].unique()}")

# Distribution by political side
print(f"\n=== DISTRIBUTION BY POLITICAL SIDE ===")
political_side_counts = cleaned_tweets_df['political_side'].value_counts()
print(political_side_counts)
for side, count in political_side_counts.items():
    percentage = (count / len(cleaned_tweets_df)) * 100
    print(f"{side}: {count} tweets ({percentage:.1f}%)")

# Distribution by party
print(f"\n=== DISTRIBUTION BY PARTY ===")
party_counts = cleaned_tweets_df['party'].value_counts()
print(party_counts)
for party, count in party_counts.items():
    percentage = (count / len(cleaned_tweets_df)) * 100
    print(f"{party}: {count} tweets ({percentage:.1f}%)")

# Top authors
print(f"\n=== TOP 10 AUTHORS BY TWEET COUNT ===")
author_counts = cleaned_tweets_df['Author'].value_counts().head(10)
for author, count in author_counts.items():
    percentage = (count / len(cleaned_tweets_df)) * 100
    party = cleaned_tweets_df[cleaned_tweets_df['Author'] == author]['party'].iloc[0]
    print(f"{author} ({party}): {count} tweets ({percentage:.1f}%)")

# Overall emotion analysis
print(f"\n=== OVERALL EMOTION ANALYSIS ===")
all_emotions = []
for emotions_str in cleaned_tweets_df['top3_emotions']:
    if emotions_str and emotions_str != '':
        try:
            emotion_names = [x.strip() for x in emotions_str.split(',')]
            all_emotions.extend(emotion_names)
        except:
            continue

emotion_counter = Counter(all_emotions)
print(f"Total emotion predictions: {len(all_emotions)}")
print(f"Most common emotions:")
for emotion_name, count in emotion_counter.most_common(15):
    emotion_id = emotion_name_to_id.get(emotion_name, 'Unknown')
    emotion_name_tr = emotion_id_to_name_tr.get(emotion_id, 'Bilinmeyen')
    percentage = (count / len(all_emotions)) * 100
    print(f"  {emotion_name} ({emotion_name_tr}) - {count} times ({percentage:.1f}%)")

# Emotion analysis by political side
print(f"\n=== EMOTION ANALYSIS BY POLITICAL SIDE ===")
# Filter out NaN values in political_side
valid_sides = cleaned_tweets_df[cleaned_tweets_df['political_side'].notna()]
for side in valid_sides['political_side'].unique():
    side_tweets = cleaned_tweets_df[cleaned_tweets_df['political_side'] == side]
    side_emotions = []
    for emotions_str in side_tweets['top3_emotions']:
        if emotions_str and emotions_str != '':
            try:
                emotion_names = [x.strip() for x in emotions_str.split(',')]
                side_emotions.extend(emotion_names)
            except:
                continue
    
    side_emotion_counter = Counter(side_emotions)
    print(f"\n{side.upper()} SIDE ({len(side_tweets)} tweets):")
    print(f"Total emotion predictions: {len(side_emotions)}")
    print("Top 10 emotions:")
    for emotion_name, count in side_emotion_counter.most_common(10):
        emotion_id = emotion_name_to_id.get(emotion_name, 'Unknown')
        emotion_name_tr = emotion_id_to_name_tr.get(emotion_id, 'Bilinmeyen')
        percentage = (count / len(side_emotions)) * 100 if side_emotions else 0
        print(f"  {emotion_name} ({emotion_name_tr}) - {count} ({percentage:.1f}%)")

# Emotion analysis by party
print(f"\n=== EMOTION ANALYSIS BY PARTY ===")
# Filter out NaN values in party
valid_parties = cleaned_tweets_df[cleaned_tweets_df['party'].notna()]
for party in valid_parties['party'].unique():
    party_tweets = cleaned_tweets_df[cleaned_tweets_df['party'] == party]
    party_emotions = []
    for emotions_str in party_tweets['top3_emotions']:
        if emotions_str and emotions_str != '':
            try:
                emotion_names = [x.strip() for x in emotions_str.split(',')]
                party_emotions.extend(emotion_names)
            except:
                continue
    
    party_emotion_counter = Counter(party_emotions)
    print(f"\n{party} PARTY ({len(party_tweets)} tweets):")
    print(f"Total emotion predictions: {len(party_emotions)}")
    print("Top 10 emotions:")
    for emotion_name, count in party_emotion_counter.most_common(10):
        emotion_id = emotion_name_to_id.get(emotion_name, 'Unknown')
        emotion_name_tr = emotion_id_to_name_tr.get(emotion_id, 'Bilinmeyen')
        percentage = (count / len(party_emotions)) * 100 if party_emotions else 0
        print(f"  {emotion_name} ({emotion_name_tr}) - {count} ({percentage:.1f}%)")

# Emotion analysis by individual authors
print(f"\n=== EMOTION ANALYSIS BY TOP AUTHORS ===")
top_authors = cleaned_tweets_df['Author'].value_counts().head(5)
for author in top_authors.index:
    author_tweets = cleaned_tweets_df[cleaned_tweets_df['Author'] == author]
    author_emotions = []
    for emotions_str in author_tweets['top3_emotions']:
        if emotions_str and emotions_str != '':
            try:
                emotion_names = [x.strip() for x in emotions_str.split(',')]
                author_emotions.extend(emotion_names)
            except:
                continue
    
    author_emotion_counter = Counter(author_emotions)
    party = author_tweets['party'].iloc[0]
    political_side = author_tweets['political_side'].iloc[0]
    
    print(f"\n{author} ({party} - {political_side}) - {len(author_tweets)} tweets:")
    print(f"Total emotion predictions: {len(author_emotions)}")
    print("Top 8 emotions:")
    for emotion_name, count in author_emotion_counter.most_common(8):
        emotion_id = emotion_name_to_id.get(emotion_name, 'Unknown')
        emotion_name_tr = emotion_id_to_name_tr.get(emotion_id, 'Bilinmeyen')
        percentage = (count / len(author_emotions)) * 100 if author_emotions else 0
        print(f"  {emotion_name} ({emotion_name_tr}) - {count} ({percentage:.1f}%)")

# Compare emotions between political sides
print(f"\n=== COMPARATIVE EMOTION ANALYSIS ===")
left_tweets = cleaned_tweets_df[cleaned_tweets_df['political_side'] == 'left']
right_tweets = cleaned_tweets_df[cleaned_tweets_df['political_side'] == 'right']

left_emotions = []
right_emotions = []

for emotions_str in left_tweets['top3_emotions']:
    if emotions_str and emotions_str != '':
        try:
            emotion_names = [x.strip() for x in emotions_str.split(',')]
            left_emotions.extend(emotion_names)
        except:
            continue

for emotions_str in right_tweets['top3_emotions']:
    if emotions_str and emotions_str != '':
        try:
            emotion_names = [x.strip() for x in emotions_str.split(',')]
            right_emotions.extend(emotion_names)
        except:
            continue

left_emotion_counter = Counter(left_emotions)
right_emotion_counter = Counter(right_emotions)

print("Emotion comparison between LEFT and RIGHT:")
all_emotion_names = set(left_emotion_counter.keys()) | set(right_emotion_counter.keys())

comparison_data = []
for emotion_name in sorted(all_emotion_names):
    left_count = left_emotion_counter.get(emotion_name, 0)
    right_count = right_emotion_counter.get(emotion_name, 0)
    left_pct = (left_count / len(left_emotions)) * 100 if left_emotions else 0
    right_pct = (right_count / len(right_emotions)) * 100 if right_emotions else 0
    diff = left_pct - right_pct
    
    emotion_id = emotion_name_to_id.get(emotion_name, 'Unknown')
    emotion_name_tr = emotion_id_to_name_tr.get(emotion_id, 'Bilinmeyen')
    
    comparison_data.append({
        'emotion_name': emotion_name,
        'emotion_name_tr': emotion_name_tr,
        'left_count': left_count,
        'right_count': right_count,
        'left_pct': left_pct,
        'right_pct': right_pct,
        'difference': diff
    })

# Sort by absolute difference
comparison_data.sort(key=lambda x: abs(x['difference']), reverse=True)

print("\nTop 15 emotions with biggest differences between LEFT and RIGHT:")
for item in comparison_data[:15]:
    print(f"{item['emotion_name']} ({item['emotion_name_tr']})")
    print(f"  LEFT: {item['left_count']} ({item['left_pct']:.1f}%) - RIGHT: {item['right_count']} ({item['right_pct']:.1f}%)")
    print(f"  Difference: {item['difference']:.1f}% {'(LEFT higher)' if item['difference'] > 0 else '(RIGHT higher)'}")
    print()

print("\n=== SENTIMENT ANALYSIS COMPLETED ===")
print(f"Analysis completed successfully!")
print(f"Dataset: {len(cleaned_tweets_df)} tweets")
print(f"Total emotion predictions: {len(all_emotions)}")
print(f"Unique emotions found: {len(emotion_counter)}")
