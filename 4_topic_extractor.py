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

# Geliştirilmiş label_map
label_map = {
    # Politik ve yönetişim
    "Göç, sığınmacı, mülteci politikaları, sınır güvenliği, entegrasyon, yabancı vatandaşların hakları ve göç yönetimi üzerine içerikler.": "migration",
    "Makroekonomi, bütçe, vergi, enflasyon, işsizlik, istihdam, para politikası ve maliye politikaları hakkında değerlendirmeler.": "economy",
    "Eğitim sistemi, okullar, üniversiteler, müfredat, öğretmen hakları, öğrenci yaşamı ve yaşam boyu öğrenme üzerine paylaşımlar.": "education",
    "Sağlık hizmetleri, hastaneler, halk sağlığı, tıbbi bakım, salgın yönetimi ve sağlık çalışanlarıyla ilgili içerikler.": "health",
    "Yargı sistemi, hukukun üstünlüğü, mahkemeler, yargı reformları, adil yargılama ve mevzuat değişiklikleri üzerine ifadeler.": "justice",
    "Dış politika, diplomasi, uluslararası anlaşmalar, bölgesel çatışmalar, küresel örgütler ve devletlerarası ilişkiler hakkında içerikler.": "foreign_policy",
    "Enerji üretimi, yenilenebilir kaynaklar, petrol, doğalgaz, enerji güvenliği, iklim değişikliği ve sürdürülebilir enerji politikaları.": "energy",
    "Yerel yönetim, belediye hizmetleri, altyapı, şehir planlama, çevre düzenlemeleri ve yerel projeler hakkında paylaşımlar.": "local_government",

    # Sosyal, kültürel ve toplumsal temalar
    "Taziye, başsağlığı, kayıp, felaket sonrası dayanışma ve acı paylaşımı ifadeleri.": "condolence",
    "Kutlama, tebrik, başarı, anma, milli bayramlar veya özel gün mesajları.": "congratulation",
    "Kültür, sanat, edebiyat, sinema, tiyatro, konserler, festivaller ve sergiler üzerine içerikler.": "culture",
    "Spor karşılaşmaları, sporcular, kulüpler, turnuvalar ve spor etkinlikleri hakkında paylaşımlar.": "sports",
    "Afetler, acil durumlar, deprem, yangın, sel gibi felaketler ve afet yönetimi üzerine içerikler.": "disaster",

    # Genel ve belirsiz
    "Belirgin bir alt alana girmeyen genel siyasi yorumlar, analizler veya gündeme dair değerlendirmeler.": "general",
    "Konu dışı, belirsiz veya sınıflandırılamayan içerikler.": "uncertain"
}

if args.topics:
    # Kullanıcı tarafından sağlanan konular varsa
    labels = [topic.strip() for topic in ' '.join(args.topics).split(',')]
    candidate_labels = labels
    label_map = None  # Kullanıcı tanımlıysa label_map yok
    print(f"Kullanıcı tanımlı konular: {labels}")
else:
    candidate_labels = list(label_map.keys())
    labels = candidate_labels
    print("Default konular kullanılıyor")

print(f"Toplam konu sayısı: {len(labels)}")

# Zero-shot sınıflandırıcıyı hazırla
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)

# Tweet veri çerçevesini yükle
df = pd.read_csv("1_politican_tweets_combined_data/all_cleaned_tweets.csv")

# Batch sınıflandırma fonksiyonu
def classify_topics_batch(texts_batch):
    results = classifier(texts_batch, candidate_labels=candidate_labels)
    topics = []
    for idx, result in enumerate(results):
        # En yüksek skor ve label
        max_score = result["scores"][0]
        long_label = result["labels"][0]
        # Skor düşükse uncertain
        if max_score < 0.3:
            topics.append("uncertain")
        else:
            # label_map varsa kısa etikete dönüştür
            short_label = label_map.get(long_label, long_label) if label_map else long_label
            topics.append(short_label)
    return topics

# Batch boyutu
batch_size = 4
topics = []

# İşlem döngüsü
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["Text"][i:i+batch_size].tolist()
    batch_topics = classify_topics_batch(batch_texts)
    topics.extend(batch_topics)

df["topic"] = topics

# Sonuçları kaydet
df.to_csv("1_politican_tweets_combined_data/all_cleaned_topic_tweets.csv", index=False)

# Konu dağılımı yazdır
print("\n=== KONU DAĞILIMI ===")
topic_counts = df["topic"].value_counts()
for topic, count in topic_counts.items():
    print(f"{topic}: {count} tweet")

print(f"\nToplam tweet sayısı: {len(df)}")

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