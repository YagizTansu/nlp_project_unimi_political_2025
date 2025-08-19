import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("1_politican_tweets_combined_data/all_cleaned_topic_tweets.csv")

# Author ve Text sütunlarına göre tekrar eden satırları sil
df = df.drop_duplicates(subset=["Author", "Text"], keep="first")

# Temizlenmiş veriyi yeni bir CSV'ye kaydet
df.to_csv("1_politican_tweets_combined_data/all_cleaned_topic_tweets.csv", index=False)

print("Tekrarlar temizlendi, '1_politican_tweets_combined_data/all_cleaned_topic_tweets.csv' dosyası oluşturuldu.")