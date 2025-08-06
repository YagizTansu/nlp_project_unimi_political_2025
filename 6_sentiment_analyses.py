import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Set Turkish locale for plots
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory for plots
plots_dir = '/home/yagiz/Desktop/nlp_project/sentiment_plots'
os.makedirs(plots_dir, exist_ok=True)

# Load emotion definitions
emotions_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/emotions_english_turkish.csv')
emotion_name_to_id = dict(zip(emotions_df['emotion_name_en'], emotions_df['emotion_id']))
emotion_id_to_name = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_en']))
emotion_id_to_name_tr = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_tr']))

# Load cleaned tweets dataset
cleaned_tweets_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/3_tweets_with_emotions/cleaned_tweets_with_emotions.csv')

# Basic dataset information
political_side_counts = cleaned_tweets_df['political_side'].value_counts()
party_counts = cleaned_tweets_df['party'].value_counts()
author_counts = cleaned_tweets_df['Author'].value_counts().head(10)

# Extract emotions
def extract_emotions(emotions_series):
    all_emotions = []
    for emotions_str in emotions_series:
        if emotions_str and isinstance(emotions_str, str) and emotions_str != '':
            try:
                emotion_names = [x.strip() for x in emotions_str.split(',')]
                all_emotions.extend(emotion_names)
            except:
                continue
    return all_emotions

# Overall emotion analysis
all_emotions = extract_emotions(cleaned_tweets_df['top3_emotions'])
emotion_counter = Counter(all_emotions)

# Create overall emotion distribution plot
plt.figure(figsize=(14, 8))
top_emotions = dict(emotion_counter.most_common(15))
emotions_df = pd.DataFrame({'Emotion': list(top_emotions.keys()), 
                           'Count': list(top_emotions.values())})
emotions_df['Turkish'] = emotions_df['Emotion'].apply(
    lambda x: emotion_id_to_name_tr.get(emotion_name_to_id.get(x, 'Unknown'), 'Bilinmeyen'))
emotions_df['Label'] = emotions_df['Emotion'] + '\n(' + emotions_df['Turkish'] + ')'
emotions_df = emotions_df.sort_values('Count', ascending=False)

sns.barplot(x='Count', y='Label', data=emotions_df, palette='viridis')
plt.title('Top 15 Emotions in All Tweets', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Emotion', fontsize=12)
plt.tight_layout()
plt.savefig(f'{plots_dir}/overall_emotion_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Political side distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=political_side_counts.index, y=political_side_counts.values, palette=['blue', 'red'])
plt.title('Tweet Distribution by Political Side', fontsize=16)
plt.xlabel('Political Side', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
for i, v in enumerate(political_side_counts.values):
    plt.text(i, v/2, f"{v} ({v/sum(political_side_counts)*100:.1f}%)", 
             ha='center', fontsize=12, color='white')
plt.tight_layout()
plt.savefig(f'{plots_dir}/political_side_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Party distribution
plt.figure(figsize=(14, 8))
party_df = pd.DataFrame({'Party': party_counts.index, 'Count': party_counts.values})
party_df = party_df.sort_values('Count', ascending=False)
sns.barplot(x='Party', y='Count', data=party_df, palette='tab10')
plt.title('Tweet Distribution by Party', fontsize=16)
plt.xlabel('Political Party', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.xticks(rotation=45)
for i, v in enumerate(party_df['Count']):
    plt.text(i, v/2, f"{v} ({v/sum(party_counts)*100:.1f}%)", 
             ha='center', fontsize=10, color='white')
plt.tight_layout()
plt.savefig(f'{plots_dir}/party_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Emotion analysis by political side
left_tweets = cleaned_tweets_df[cleaned_tweets_df['political_side'] == 'left']
right_tweets = cleaned_tweets_df[cleaned_tweets_df['political_side'] == 'right']

left_emotions = extract_emotions(left_tweets['top3_emotions'])
right_emotions = extract_emotions(right_tweets['top3_emotions'])

left_emotion_counter = Counter(left_emotions)
right_emotion_counter = Counter(right_emotions)

# Create comparison dataframe
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

# Sort by absolute difference and get top 15
comparison_data.sort(key=lambda x: abs(x['difference']), reverse=True)
top_comparison = pd.DataFrame(comparison_data[:15])

# Create left vs right emotion comparison plot
plt.figure(figsize=(14, 10))
top_comparison['Label'] = top_comparison['emotion_name'] + '\n(' + top_comparison['emotion_name_tr'] + ')'
plt.barh(top_comparison['Label'], top_comparison['left_pct'], color='blue', alpha=0.7, label='Left')
plt.barh(top_comparison['Label'], -top_comparison['right_pct'], color='red', alpha=0.7, label='Right')
plt.axvline(x=0, color='black', linestyle='-')
plt.xlabel('Percentage (%)', fontsize=12)
plt.title('Top 15 Emotions with Biggest Differences Between Left and Right', fontsize=16)
plt.legend()

# Add percentage labels
for i, row in enumerate(top_comparison.itertuples()):
    plt.text(row.left_pct+0.5, i, f"{row.left_pct:.1f}%", va='center')
    plt.text(-row.right_pct-3.5, i, f"{row.right_pct:.1f}%", va='center')

plt.tight_layout()
plt.savefig(f'{plots_dir}/left_vs_right_emotions.png', dpi=300, bbox_inches='tight')
plt.close()

# Emotion analysis by party
plt.figure(figsize=(16, 12))

# Filter out NaN values in party
valid_parties = cleaned_tweets_df[cleaned_tweets_df['party'].notna()]

# Get top 8 emotions across all parties
all_party_emotions = []
for party in valid_parties['party'].unique():
    party_emotions = extract_emotions(cleaned_tweets_df[cleaned_tweets_df['party'] == party]['top3_emotions'])
    all_party_emotions.extend(party_emotions)

top_emotions_overall = [e for e, _ in Counter(all_party_emotions).most_common(8)]

# Create heatmap data
party_emotion_data = []
for party in valid_parties['party'].unique():
    party_tweets = cleaned_tweets_df[cleaned_tweets_df['party'] == party]
    party_emotions = extract_emotions(party_tweets['top3_emotions'])
    party_emotion_counter = Counter(party_emotions)
    
    for emotion in top_emotions_overall:
        count = party_emotion_counter.get(emotion, 0)
        percentage = (count / len(party_emotions)) * 100 if party_emotions else 0
        party_emotion_data.append({
            'Party': party,
            'Emotion': emotion,
            'Percentage': percentage
        })

emotion_heatmap_df = pd.DataFrame(party_emotion_data)
emotion_heatmap_pivot = emotion_heatmap_df.pivot(index='Party', columns='Emotion', values='Percentage')

# Create heatmap
sns.heatmap(emotion_heatmap_pivot, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5)
plt.title('Emotion Distribution by Political Party (Percentage)', fontsize=16)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Party', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{plots_dir}/party_emotion_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Emotion analysis for top authors
top_5_authors = author_counts.index[:5]
author_emotion_data = []

for author in top_5_authors:
    author_tweets = cleaned_tweets_df[cleaned_tweets_df['Author'] == author]
    author_emotions = extract_emotions(author_tweets['top3_emotions'])
    author_emotion_counter = Counter(author_emotions)
    top_8_emotions = [e for e, _ in author_emotion_counter.most_common(8)]
    
    party = author_tweets['party'].iloc[0]
    political_side = author_tweets['political_side'].iloc[0]
    
    # Create radar chart data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Get emotion percentages
    emotions = []
    percentages = []
    for emotion in top_8_emotions:
        emotions.append(emotion)
        percentage = (author_emotion_counter[emotion] / len(author_emotions)) * 100
        percentages.append(percentage)
    
    # Number of variables
    N = len(emotions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    percentages += percentages[:1]  # Close the loop
    
    # Draw polygon and fill it
    ax.plot(angles, percentages, linewidth=1, linestyle='solid')
    ax.fill(angles, percentages, alpha=0.1)
    
    # Add labels
    plt.xticks(angles[:-1], emotions, size=10)
    plt.yticks(np.arange(0, max(percentages)+10, 10), size=8)
    
    plt.title(f'Emotion Profile for {author}\n({party} - {political_side})', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/author_{author.replace(" ", "_")}_emotions.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create a dashboard with subplots
plt.figure(figsize=(20, 15))
gs = GridSpec(3, 2, figure=plt.gcf())

# Political sides distribution
ax1 = plt.subplot(gs[0, 0])
sns.barplot(x=political_side_counts.index, y=political_side_counts.values, palette=['blue', 'red'], ax=ax1)
ax1.set_title('Tweet Distribution by Political Side', fontsize=14)
ax1.set_xlabel('Political Side', fontsize=10)
ax1.set_ylabel('Number of Tweets', fontsize=10)
for i, v in enumerate(political_side_counts.values):
    ax1.text(i, v/2, f"{v} ({v/sum(political_side_counts)*100:.1f}%)", 
             ha='center', fontsize=10, color='white')

# Overall emotion distribution
ax2 = plt.subplot(gs[0, 1])
top_emotions = dict(emotion_counter.most_common(10))
emotions_df = pd.DataFrame({'Emotion': list(top_emotions.keys()), 
                           'Count': list(top_emotions.values())})
emotions_df['Label'] = emotions_df['Emotion']
emotions_df = emotions_df.sort_values('Count', ascending=True)
sns.barplot(x='Count', y='Label', data=emotions_df, palette='viridis', ax=ax2)
ax2.set_title('Top 10 Emotions in All Tweets', fontsize=14)
ax2.set_xlabel('Count', fontsize=10)
ax2.set_ylabel('Emotion', fontsize=10)

# Left vs Right emotion comparison
ax3 = plt.subplot(gs[1, :])
top_8_comparison = pd.DataFrame(comparison_data[:8])
top_8_comparison['Label'] = top_8_comparison['emotion_name']
ax3.barh(top_8_comparison['Label'], top_8_comparison['left_pct'], color='blue', alpha=0.7, label='Left')
ax3.barh(top_8_comparison['Label'], -top_8_comparison['right_pct'], color='red', alpha=0.7, label='Right')
ax3.axvline(x=0, color='black', linestyle='-')
ax3.set_xlabel('Percentage (%)', fontsize=10)
ax3.set_title('Emotions with Biggest Differences Between Left and Right', fontsize=14)
ax3.legend()

# Party emotion heatmap
ax4 = plt.subplot(gs[2, :])
sns.heatmap(emotion_heatmap_pivot.iloc[:, :6], annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5, ax=ax4)
ax4.set_title('Emotion Distribution by Political Party (Percentage)', fontsize=14)
ax4.set_xlabel('Emotion', fontsize=10)
ax4.set_ylabel('Party', fontsize=10)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{plots_dir}/sentiment_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# Create party comparison bar charts
top_parties = party_counts.head(6).index

plt.figure(figsize=(18, 12))
for i, emotion in enumerate(top_emotions_overall[:6]):
    plt.subplot(2, 3, i+1)
    emotion_by_party = []
    
    for party in top_parties:
        party_tweets = cleaned_tweets_df[cleaned_tweets_df['party'] == party]
        party_emotions = extract_emotions(party_tweets['top3_emotions'])
        party_emotion_counter = Counter(party_emotions)
        emotion_count = party_emotion_counter.get(emotion, 0)
        percentage = (emotion_count / len(party_emotions)) * 100 if party_emotions else 0
        emotion_by_party.append({
            'Party': party,
            'Percentage': percentage
        })
    
    emotion_party_df = pd.DataFrame(emotion_by_party)
    sns.barplot(x='Party', y='Percentage', data=emotion_party_df)
    plt.title(f'"{emotion}" by Party', fontsize=12)
    plt.xlabel('Party', fontsize=10)
    plt.ylabel('Percentage (%)', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    
plt.tight_layout()
plt.savefig(f'{plots_dir}/emotion_by_party_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
