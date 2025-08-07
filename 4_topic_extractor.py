import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import argparse

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
    # Default konuları kullan
    labels = [
        "göç",            # sığınmacılar, mülteciler, sınır güvenliği
        "ekonomi",        # enflasyon, işsizlik, zam, maaş
        "eğitim",         # öğretmen, okul, sınav, üniversite
        "sağlık",         # doktor, hastane, aşı, pandemi
        "adalet",         # yargı, mahkeme, hukuk sistemi
        "dış politika",   # NATO, AB, ABD, büyükelçi
        "enerji",         # doğalgaz, elektrik, yenilenebilir
        "başsağlığı",     # başsağlığı, geçmiş olsun, taziye
        "genel"           # hiçbirine doğrudan girmeyenler
    ]
    print("Default konular kullanılıyor")

print(f"Toplam konu sayısı: {len(labels)}")

classifier = pipeline("zero-shot-classification", model="dbmdz/bert-base-turkish-cased", device=device)

# Tweet veri çerçevesi
df = pd.read_csv("1_politican_tweets_combined_data/all_cleaned_tweets.csv")

# Batch sınıflandırma fonksiyonu
def classify_topics_batch(texts_batch):
    results = classifier(texts_batch, candidate_labels=labels)
    return [result["labels"][0] for result in results]

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