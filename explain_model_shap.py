import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from deep_translator import GoogleTranslator

# -----------------------------
# Device & model setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Future-proof attention implementation
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    attn_implementation="eager"  # avoids warning in transformers v5.0.0+
)
model.to(device)
model.eval()

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 0
if not hasattr(model.config, "pad_token_id") or model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# -----------------------------
# Pipeline for SHAP explainability
# -----------------------------
nlp_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# -----------------------------
# Load dataset (5 tweets per emotion)
# -----------------------------
df = pd.read_csv("./outputs/all_cleaned_tweets_with_topics_and_emotions.csv")
target_emotions = ["Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise"]
emotion_samples = []
for emotion in target_emotions:
    samples = df[df["predicted_emotions"] == emotion].head(5)
    emotion_samples.append(samples)
df_selected = pd.concat(emotion_samples, ignore_index=True)
tweet_texts = df_selected["Text"].tolist()
tweet_emotions = df_selected["predicted_emotions"].tolist()

# Translate each tweet to English using deep-translator
tweet_texts_en = []
for text in tweet_texts:
    try:
        translation = GoogleTranslator(source='auto', target='en').translate(text)
        tweet_texts_en.append(translation)
    except Exception as e:
        tweet_texts_en.append("[Translation error]")

# Convert to HuggingFace Dataset for efficient SHAP
hf_dataset = Dataset.from_pandas(df_selected[["Text"]])

# -----------------------------
# SHAP explainability (batched)
# -----------------------------
explainer = shap.Explainer(nlp_pipe)
shap_values = explainer(hf_dataset["Text"])  # batched explanations

# Save all SHAP explanations in a single modern HTML file
all_html_blocks = []
sidebar_links = []

# Group tweets by emotion for sidebar and blocks
emotion_indices = {emotion: [] for emotion in target_emotions}
for i, emotion in enumerate(tweet_emotions):
    emotion_indices[emotion].append(i)

for emotion in target_emotions:
    sidebar_links.append(f'<div class="emotion-group"><strong>{emotion}</strong></div>')
    for idx in emotion_indices[emotion]:
        block_id = f"{emotion.lower()}-{idx+1}"
        sidebar_links.append(f'<a href="#{block_id}">{emotion} Tweet {idx+1-emotion_indices[emotion][0]+1}</a>')

for i, (tweet, tweet_en, emotion) in enumerate(zip(tweet_texts, tweet_texts_en, tweet_emotions)):
    shap_html = shap.plots.text(shap_values[i], display=False)
    block_id = f"{emotion.lower()}-{i+1}"
    all_html_blocks.append(f"""
    <section class="explanation-block" id="{block_id}">
        <div class="tweet-header">
            <span class="tweet-title">{emotion} Tweet {i+1-emotion_indices[emotion][0]+1}</span>
            <button class="toggle-btn" onclick="toggleBlock('{block_id}-content')">Show/Hide</button>
        </div>
        <div class="tweet-text">{tweet}</div>
        <div class="tweet-text" style="background:#eaf7e1;color:#1a5e1a;">
            <strong>English:</strong> {tweet_en}
        </div>
        <div class="shap-content" id="{block_id}-content">
            {shap_html}
        </div>
    </section>
    """)

main_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SHAP Explanations for All Emotions</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f4f6fb;
            margin: 0;
            color: #222;
        }}
        .layout {{
            display: flex;
            min-height: 100vh;
        }}
        .sidebar {{
            background: #2a4d8f;
            color: #fff;
            width: 220px;
            padding: 32px 18px;
            box-sizing: border-box;
            flex-shrink: 0;
        }}
        .sidebar h2 {{
            font-size: 1.2em;
            margin-bottom: 18px;
            letter-spacing: 1px;
        }}
        .sidebar a {{
            display: block;
            color: #fff;
            text-decoration: none;
            margin-bottom: 12px;
            padding: 8px 0;
            border-radius: 4px;
            transition: background 0.2s;
        }}
        .sidebar a:hover {{
            background: #1d3557;
        }}
        .content {{
            flex: 1;
            padding: 40px 5vw 40px 5vw;
            background: #f4f6fb;
        }}
        .explanation-block {{
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            margin-bottom: 36px;
            padding: 28px 24px 18px 24px;
        }}
        .tweet-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .tweet-title {{
            font-size: 1.15em;
            font-weight: 600;
            color: #2a4d8f;
        }}
        .toggle-btn {{
            background: #2a4d8f;
            color: #fff;
            border: none;
            border-radius: 4px;
            padding: 6px 14px;
            cursor: pointer;
            font-size: 0.95em;
            transition: background 0.2s;
        }}
        .toggle-btn:hover {{
            background: #1d3557;
        }}
        .tweet-text {{
            font-size: 1.08em;
            margin-bottom: 18px;
            color: #222;
            background: #f0f4fa;
            padding: 10px 16px;
            border-radius: 6px;
        }}
        .emotion-group {{
            margin-top: 18px;
            margin-bottom: 6px;
            font-weight: bold;
            color: #ffd700;
            font-size: 1.08em;
        }}
        @media (max-width: 800px) {{
            .layout {{
                flex-direction: column;
            }}
            .sidebar {{
                width: 100%;
                padding: 18px 8px;
            }}
            .content {{
                padding: 18px 2vw 18px 2vw;
            }}
        }}
    </style>
    <script>
        function toggleBlock(id) {{
            var el = document.getElementById(id);
            if (el.style.display === "none") {{
                el.style.display = "block";
            }} else {{
                el.style.display = "none";
            }}
        }}
        window.onload = function() {{
            // Hide all SHAP blocks except the first
            {''.join([f'document.getElementById("{tweet_emotions[i].lower()}-{i+1}-content").style.display = "{ "block" if i==0 else "none" }";' for i in range(len(tweet_texts))])}
        }};
    </script>
</head>
<body>
    <div class="layout">
        <nav class="sidebar">
            <h2>Emotions</h2>
            {''.join(sidebar_links)}
        </nav>
        <main class="content">
            <h1 style="color:#2a4d8f; margin-bottom:32px;">SHAP Explanations for All Emotions</h1>
            {''.join(all_html_blocks)}
        </main>
    </div>
</body>
</html>
"""


def plot_attention_heatmap(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # list of layers
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # filter out special tokens
    token_indices = [i for i, tok in enumerate(tokens) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]
    filtered_tokens = [tokens[i] for i in token_indices]

    attn = attentions[-1][0][0].cpu().numpy()  # last layer, head 0
    filtered_attn = attn[np.ix_(token_indices, token_indices)]

    plt.figure(figsize=(max(6, len(filtered_tokens)), 6))
    im = plt.imshow(filtered_attn, cmap="viridis", interpolation="nearest")
    plt.xticks(range(len(filtered_tokens)), filtered_tokens, rotation=90)
    plt.yticks(range(len(filtered_tokens)), filtered_tokens)
    plt.title("Attention Heatmap (Last Layer, Head 0)")
    plt.xlabel("Input Tokens")
    plt.ylabel("Attended Tokens")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


out_path = "visualizations/shap_explanations_all.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(main_html)
print(f"Saved all SHAP explanations to → {out_path}")

# # Plot attention heatmap for one tweet from each emotion
# for emotion in target_emotions:
#     idx = emotion_indices[emotion][0]  # first tweet index for this emotion
#     print(f"Plotting attention heatmap for {emotion} (tweet index {idx})")
#     plot_attention_heatmap(model, tokenizer, tweet_texts[idx])