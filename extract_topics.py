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

# Türkçe -> İngilizce konu eşlemesi
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

# Varsayılan (MODEL İÇİN) Türkçe konu etiketleri
default_candidate_labels = list(label_map.keys())

# Komut satırı argümanlarını parse et
parser = argparse.ArgumentParser(description='Tweet konu sınıflandırması')
parser.add_argument('--topics', nargs='*', help='Kullanıcı tanımlı konular (Türkçe veya İngilizce, virgülle ayrılmış)')
args = parser.parse_args()

# Kullanıcı tanımlı konular varsa işle (Türkçe veya İngilizce kabul)
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
            print(f"Uyarı: '{t}' tanınmadı ve atlandı.")
    if not candidate_labels:
        print("Geçerli konu bulunamadı, varsayılanlar kullanılıyor.")
        candidate_labels = default_candidate_labels
    print(f"Modelde kullanılacak (Türkçe) konular: {candidate_labels}")
else:
    candidate_labels = default_candidate_labels
    print("Varsayılan Türkçe konular kullanılıyor.")

print(f"Toplam konu sayısı (model): {len(candidate_labels)}")

# Zero-shot sınıflandırıcıyı hazırla
classifier = pipeline("zero-shot-classification", 
                        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 
                        device=device,multi_label=True)

# Tweet veri çerçevesini yükle
df = pd.read_csv("data_processed/all_cleaned_tweets.csv")

# Batch sınıflandırma fonksiyonu (Türkçe etiket -> İngilizce kaydet)
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

# Batch boyutu
batch_size = 8
topics = []

# İşlem döngüsü
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["Text"][i:i+batch_size].tolist()
    batch_topics = classify_topics_batch(batch_texts)
    topics.extend(batch_topics)

# Sonuçlar İngilizce etiketlerle kaydediliyor
df["topic"] = topics
df.to_csv("data_processed/all_cleaned_tweets_with_topics.csv", index=False)

# Konu dağılımı (İngilizce)
print("\n=== KONU DAĞILIMI (İngilizce) ===")
topic_counts = df["topic"].value_counts()
for topic, count in topic_counts.items():
    print(f"{topic}: {count} tweet")

# Her konudan 5'er örnek yazdır
print("\n=== HER KONUDAN 5 RASTGELE ÖRNEK ===")
for topic in topic_counts.index:
    topic_tweets = df[df["topic"] == topic]["Text"].tolist()
    if len(topic_tweets) > 0:
        sample_size = min(5, len(topic_tweets))
        random_samples = random.sample(topic_tweets, sample_size)
        
        print(f"\n--- {topic.upper()} ({sample_size} örnek) ---")
        for i, tweet in enumerate(random_samples, 1):
            print(f"{i}. {tweet[:200]}{'...' if len(tweet) > 200 else ''}")