import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import argparse
import random

# tqdm'u pandas ile entegre et
tqdm.pandas()

# GPU kontrolü
device = 0 if torch.cuda.is_available() else -1
print(f"Kullanılan cihaz: {'GPU' if device == 0 else 'CPU'}")

# Komut satırı argümanlarını parse et
parser = argparse.ArgumentParser(description='Tweet konu sınıflandırması')
parser.add_argument('--topics', nargs='*', help='Sınıflandırma için kullanılacak konular (virgülle ayrılmış)')
args = parser.parse_args()

# Konuları belirle
if args.topics:
    # Komut satırından gelen konuları kullan
    labels = [topic.strip() for topic in ' '.join(args.topics).split(',')]
    print(f"Kullanıcı tanımlı konular: {labels}")
else:
    # Default konuları kullan - zero-shot model için cümle formatında etiketler
    labels = [
        "Bu tweet göç ve mülteci politikaları ile ilgilidir.",
        "Bu tweet ekonomi ve mali politikalarla ilgilidir.",
        "Bu tweet eğitim ve öğretim konularını içermektedir.",
        "Bu tweet sağlık ve tıp konuları ile ilgilidir.",
        "Bu tweet adalet ve hukuk sistemi ile ilgilidir.",
        "Bu tweet dış politika ve uluslararası ilişkilerle ilgilidir.",
        "Bu tweet enerji ve doğal kaynaklarla ilgilidir.",
        "Bu tweet taziye ve başsağlığı mesajları içermektedir.",
        "Bu tweet tebrik ve kutlama mesajları içermektedir.",
        "Bu tweet belirli bir konu içermeyen siyasi tweet içermektedir.",
    ]
    print("Default konular kullanılıyor")

print(f"Toplam konu sayısı: {len(labels)}")

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)

# Tweet veri çerçevesi
df = pd.read_csv("1_politican_tweets_combined_data/all_cleaned_tweets.csv")

# Batch sınıflandırma fonksiyonu
def classify_topics_batch(texts_batch):
    results = classifier(texts_batch, candidate_labels=labels)
    topics = []
    for result in results:
        if result["scores"][0] < 0.1:
            topic = "belirsiz"
        else:
            topic = result["labels"][0]
        topics.append(topic)
    return topics

# Batch işleme
batch_size = 16
topics = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["Text"][i:i+batch_size].tolist()
    batch_topics = classify_topics_batch(batch_texts)
    topics.extend(batch_topics)

df["topic"] = topics

# Sonuçları kaydet
df.to_csv("1_politican_tweets_combined_data/all_cleaned_topic_tweets.csv", index=False)

# Konu dağılımını yazdır
print("\n=== KONU DAĞILIMI ===")
topic_counts = df["topic"].value_counts()
for topic, count in topic_counts.items():
    print(f"{topic}: {count} tweet")

print(f"\nToplam tweet sayısı: {len(df)}")

# Her konudan 5'er örnek bastır
print("\n=== HER KONUDAN 5 RASTGELE ÖRNEK ===")

for topic in topic_counts.index:
    topic_tweets = df[df["topic"] == topic]["Text"].tolist()
    if len(topic_tweets) > 0:
        sample_size = min(5, len(topic_tweets))
        random_samples = random.sample(topic_tweets, sample_size)
        
        print(f"\n--- {topic.upper()} ({sample_size} örnek) ---")
        for i, tweet in enumerate(random_samples, 1):
            print(f"{i}. {tweet[:200]}{'...' if len(tweet) > 200 else ''}")