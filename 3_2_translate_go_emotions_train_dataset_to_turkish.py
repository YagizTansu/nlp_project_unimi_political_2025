import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch
import gc

# GPU kullanımını kontrol et
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model ve tokenizer'ı yükle
model_name = 'Helsinki-NLP/opus-mt-tc-big-en-tr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# CSV dosyasını oku (örnekle)
df = pd.read_csv("/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/go_emotions_english_train.csv", header=None)
df.columns = ['text', 'col2', 'col3']

# Batch çeviri fonksiyonu
def translate_batch(texts, batch_size=2):  # Batch size daha da düşük
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch çevirisi"):
        try:
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize et (daha kısa max_length)
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Çeviri yap (beam search yerine greedy decoding)
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=256, do_sample=False, early_stopping=True)
            
            # Decode et
            batch_results = tokenizer.batch_decode(translated, skip_special_tokens=True)
            results.extend(batch_results)
            
            # Her batch sonrası memory temizle
            del inputs, translated
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM error at batch {i}, trying with smaller batch...")
            # Tek tek çevir
            for text in batch_texts:
                try:
                    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        translated = model.generate(**inputs, max_length=256, do_sample=False)
                    
                    result = tokenizer.decode(translated[0], skip_special_tokens=True)
                    results.append(result)
                    
                    del inputs, translated
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    print(f"Error translating text: {e}")
                    results.append(text)  # Fallback: orijinal metni kullan
    
    return results

# Çeviri işlemini başlat
print("Batch çeviri işlemi başlıyor...")
df['text'] = translate_batch(df['text'].tolist(), batch_size=8)  # İngilizce yerine Türkçe yaz
print("Çeviri işlemi tamamlandı!")

# Yeni CSV dosyasına kaydet (aynı format, sadece text sütunu Türkçe)
df.to_csv("/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/go_emotions_turkish_train.csv", index=False, header=False)

print("Dosya kaydedildi!")
