import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import json

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_turkish_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load label mappings
with open(f'{model_path}/label_mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings['id2label'].items()}
    label2id = mappings['label2id']
    emotion_labels = mappings['emotion_labels']

print(f"Loaded fine-tuned model from: {model_path}")
print(f"Available emotions: {emotion_labels}")

# Prediction function for single-class classification
def predict_emotion(text, return_top_k=3):
    model.eval()
    if not text.strip():
        return None, []
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # Get top predictions
    top_indices = probabilities.argsort()[-return_top_k:][::-1]
    top_predictions = [(id2label[idx], probabilities[idx]) for idx in top_indices]
    
    # Best prediction
    best_emotion = id2label[probabilities.argmax()]
    
    return best_emotion, top_predictions

# Test predictions
sample_text = "Bu gerçekten harika bir gün!"
predicted_emotion, top_predictions = predict_emotion(sample_text)

print(f"\nSample prediction for: '{sample_text}'")
print(f"Predicted emotion: {predicted_emotion}")
print("Top 3 predictions:")
for emotion, score in top_predictions:
    print(f"  {emotion}: {score:.4f}")

# Batch test
sample_political_tweet_texts = [
    "Bugün çok üzgünüm, ülkemiz için endişeliyim.",
    "Bu seçim sonuçları beni çok mutlu etti!",
    "Hükümetin politikalarını eleştiriyorum, daha iyi bir gelecek istiyorum.",
    "Bu konuda kafam çok karışık, ne yapacağımı bilmiyorum.",
    "Partim için çok heyecanlıyım, yeni projelerimiz var!",
    "Bu tweeti yazarken çok kızgınım, adaletsizliklere karşıyım.",
    "Sevgi dolu bir toplum için çalışmalıyız, birlik olmalıyız.",
    "Bu konuda çok şaşkınım, beklemediğim bir durumla karşılaştım.",
    "Ülkemizin geleceği için umutluyum, gençlerimize güveniyorum.",
    "Bu konuda çok endişeliyim, geleceğimiz tehlikede.",
    "Bu tweeti yazarken çok mutluyum, güzel bir haber aldım.",
    "Bu konuda çok kızgınım, adaletsizliklere karşıyım.",
    "Cezaevinden yeni çıkan eski Eşbaşkanımız Fikret Doğanı Vanda evinde ziyaret ettik. Tekrar geçmiş olsun.",
    "Her şey onların hayaliyle başladı. Trabzonsporumuzun temellerini atan kurucularımızı saygı, sevgi ve minnetle anıyoruz."
]

print("\nTesting model with multiple samples:")
for text in sample_political_tweet_texts:
    pred_emotion, top_preds = predict_emotion(text)
    print(f"\nText: {text}")
    print(f"Predicted emotion: {pred_emotion}")
    print("Top 3 predictions:")
    for emotion, score in top_preds:
        print(f"  {emotion}: {score:.4f}")

print("\nTesting completed successfully!")

# Process all_cleaned_tweets.csv file
print("\nProcessing all_cleaned_tweets.csv...")

# Load the full tweets dataset
full_tweets_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/1_politican_tweets_combined_data/all_cleaned_topic_tweets.csv')
print(f"Loaded {len(full_tweets_df)} tweets")

# Function to get emotion prediction for a text
def get_emotion_prediction(text):
    try:
        if not text.strip():
            return "", ""
        
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # Get emotions with confidence over 20%
        high_confidence_indices = [idx for idx, prob in enumerate(probabilities) if prob >= 0.05]
        
        # Sort by confidence (highest first) and take top 2
        high_confidence_indices = sorted(high_confidence_indices, key=lambda idx: probabilities[idx], reverse=True)[:2]
        
        # If no emotions have 20%+ confidence, just take the top one
        if not high_confidence_indices:
            high_confidence_indices = [probabilities.argmax()]
            
        # Format top emotions as comma-separated string
        top_emotions = [id2label[idx] for idx in high_confidence_indices]
        top_emotions_str = ",".join(top_emotions)
        
        # Get top 3 predictions as comma-separated string (keep this for the top3_emotions column)
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_emotions = [id2label[idx] for idx in top_3_indices]
        top_3_str = ",".join(top_3_emotions)
        
        return top_emotions_str, top_3_str
    
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return "", ""

# Apply prediction to each row
text_col = 'Text' if 'Text' in full_tweets_df.columns else full_tweets_df.columns[0]
print(f"Using column '{text_col}' for predictions")

# Clean texts and create new dataframe with only necessary columns
print("Preparing data...")
cleaned_tweets_df = full_tweets_df[['ID', 'Author', 'party', 'political_side', 'Date','topic', text_col]].copy()

# Reset index
cleaned_tweets_df = cleaned_tweets_df.reset_index(drop=True)

# Process in batches for better performance
batch_size = 100
predicted_emotions = []
top3_emotions = []

print("Processing emotion predictions...")
for i in range(0, len(cleaned_tweets_df), batch_size):
    batch_end = min(i + batch_size, len(cleaned_tweets_df))
    batch_texts = cleaned_tweets_df[text_col].iloc[i:batch_end]
    
    batch_predictions = []
    batch_top3 = []
    for text in batch_texts:
        emotion, top3 = get_emotion_prediction(text)
        batch_predictions.append(emotion)
        batch_top3.append(top3)
    
    predicted_emotions.extend(batch_predictions)
    top3_emotions.extend(batch_top3)
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {batch_end}/{len(cleaned_tweets_df)} tweets...")

# Add the emotion prediction columns
cleaned_tweets_df['predicted_emotions'] = predicted_emotions
cleaned_tweets_df['top3_emotions'] = top3_emotions


# Save the cleaned dataset
output_path = '/home/yagiz/Desktop/nlp_project/3_tweets_with_emotions/all_cleaned_tweets_with_topics_and_emotions.csv'
cleaned_tweets_df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")

# Print emotion distribution
print(f"\nEmotion distribution in the dataset:")
emotion_counts = pd.Series([emotion for emotions in predicted_emotions for emotion in emotions.split(',')]).value_counts()
print(emotion_counts)