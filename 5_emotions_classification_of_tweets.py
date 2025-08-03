import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_turkish_bert_emotions"
print(f"Loading model from: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you've run the fine-tuning script first (4_fine_tune_bert_model.py)")
    exit(1)

# Load emotion labels mapping
emotions_df = pd.read_csv('turkish_emotions_datasets/emotions_english_turkish.csv')
emotion_id_to_name = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_en']))
print(f"Loaded {len(emotion_id_to_name)} emotion labels")

# Load tweet dataset
print("Loading tweet dataset...")
tweets_df = pd.read_csv('politican_tweets_combined_data/full_tweets.csv')
print(f"Loaded {len(tweets_df)} tweets")

# Display dataset info
print(f"Dataset columns: {list(tweets_df.columns)}")
print(f"Authors: {tweets_df['Author'].nunique()}")
print(f"Parties: {tweets_df['party'].value_counts().to_dict()}")
print(f"Political sides: {tweets_df['political_side'].value_counts().to_dict()}")

def predict_emotions_batch(texts, threshold=0.3, batch_size=32):
    """Predict emotions for a batch of texts"""
    all_predictions = []
    all_scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
        
        # Process each text in batch
        for pred in predictions:
            # Get predicted emotion IDs above threshold
            predicted_emotions = [i for i, score in enumerate(pred) if score > threshold]
            all_predictions.append(predicted_emotions)
            all_scores.append(pred)
    
    return all_predictions, all_scores

def format_emotions(emotion_ids, scores=None, top_k=3):
    """Format emotion predictions for display"""
    if not emotion_ids:
        return "neutral", ""
    
    # Get emotion names
    emotion_names = [emotion_id_to_name.get(eid, f"unknown_{eid}") for eid in emotion_ids]
    
    if scores is not None:
        # Sort by scores and get top emotions
        emotion_score_pairs = [(eid, scores[eid]) for eid in emotion_ids]
        emotion_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_emotions = emotion_score_pairs[:top_k]
        primary_emotion = emotion_id_to_name.get(top_emotions[0][0], "unknown")
        
        # Create detailed string with scores
        detailed = "; ".join([f"{emotion_id_to_name.get(eid, f'unk_{eid}')}:{score:.3f}" 
                             for eid, score in top_emotions])
        
        return primary_emotion, detailed
    else:
        return emotion_names[0] if emotion_names else "neutral", "; ".join(emotion_names)

# Apply emotion classification
print("Applying emotion classification to tweets...")
tweet_texts = tweets_df['Text'].fillna("").tolist()

# Predict emotions
predicted_emotions, emotion_scores = predict_emotions_batch(tweet_texts, threshold=0.3)

# Process results
print("Processing results...")
primary_emotions = []
detailed_emotions = []
emotion_counts = []

for i, (emotions, scores) in enumerate(zip(predicted_emotions, emotion_scores)):
    primary, detailed = format_emotions(emotions, scores, top_k=3)
    primary_emotions.append(primary)
    detailed_emotions.append(detailed)
    emotion_counts.append(len(emotions))

# Add results to dataframe
tweets_df['primary_emotion'] = primary_emotions
tweets_df['detailed_emotions'] = detailed_emotions
tweets_df['emotion_count'] = emotion_counts

# Add individual emotion scores as binary columns (optional)
print("Adding individual emotion columns...")
for emotion_id, emotion_name in emotion_id_to_name.items():
    emotion_binary = [1 if emotion_id in pred else 0 for pred in predicted_emotions]
    tweets_df[f'emotion_{emotion_name}'] = emotion_binary

# Save results
output_file = 'politican_tweets_combined_data/tweets_with_emotions.csv'
tweets_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Generate analysis report
print("\n" + "="*50)
print("EMOTION ANALYSIS REPORT")
print("="*50)

# Overall statistics
print(f"\nDataset Overview:")
print(f"Total tweets: {len(tweets_df)}")
print(f"Average emotions per tweet: {np.mean(emotion_counts):.2f}")
print(f"Tweets with no emotions: {sum(1 for x in emotion_counts if x == 0)}")

# Most common primary emotions
print(f"\nTop 10 Primary Emotions:")
emotion_dist = tweets_df['primary_emotion'].value_counts()
for emotion, count in emotion_dist.head(10).items():
    percentage = (count / len(tweets_df)) * 100
    print(f"  {emotion}: {count} ({percentage:.1f}%)")

# Emotions by political side
print(f"\nEmotions by Political Side:")
for side in tweets_df['political_side'].unique():
    side_tweets = tweets_df[tweets_df['political_side'] == side]
    top_emotion = side_tweets['primary_emotion'].value_counts().head(3)
    print(f"  {side.upper()}:")
    for emotion, count in top_emotion.items():
        percentage = (count / len(side_tweets)) * 100
        print(f"    {emotion}: {count} ({percentage:.1f}%)")

# Emotions by party
print(f"\nEmotions by Party:")
for party in tweets_df['party'].unique():
    party_tweets = tweets_df[tweets_df['party'] == party]
    top_emotion = party_tweets['primary_emotion'].value_counts().head(2)
    print(f"  {party}:")
    for emotion, count in top_emotion.items():
        percentage = (count / len(party_tweets)) * 100
        print(f"    {emotion}: {count} ({percentage:.1f}%)")

# Most emotional authors
print(f"\nMost Emotional Authors (avg emotions per tweet):")
author_emotion_avg = tweets_df.groupby('Author')['emotion_count'].mean().sort_values(ascending=False)
for author, avg_emotions in author_emotion_avg.head(5).items():
    tweet_count = len(tweets_df[tweets_df['Author'] == author])
    party = tweets_df[tweets_df['Author'] == author]['party'].iloc[0]
    print(f"  {author} ({party}): {avg_emotions:.2f} emotions/tweet ({tweet_count} tweets)")

# Sample tweets with high emotion scores
print(f"\nSample Highly Emotional Tweets:")
high_emotion_tweets = tweets_df[tweets_df['emotion_count'] >= 3].head(3)
for idx, row in high_emotion_tweets.iterrows():
    print(f"\n  Author: {row['Author']} ({row['party']})")
    print(f"  Text: {row['Text'][:100]}...")
    print(f"  Emotions: {row['detailed_emotions']}")

print(f"\n" + "="*50)
print("Analysis completed successfully!")
print(f"Full results saved to: {output_file}")
