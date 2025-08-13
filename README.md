# NLP Project - Political Tweet Analysis

Repository containing the code of the project for the "Natural Language Processing" course (Academic Year 2024-25) at the University of Milan, as part of the Master Degree in Computer Science.

**This project is intended for educational purposes only.**

This project collects and analyzes tweets from Turkish politicians to perform natural language processing tasks.

## Project Proposal

**Politics of emotions or propaganda? (P3)**

This project explores how emotional language is used strategically in political texts, such as speeches, social media posts, or debates—to influence perception and manipulate audience response. Students will design a pipeline using transformer-based models to detect emotional framing, categorize tone (e.g., fear, pride, outrage), and highlight shifts in sentiment across political stances or media sources. The goal is not just to classify emotion, but to interpret its rhetorical function within the discourse. 

In order to perform the task, the project should:

- Use pre-trained transformer models (e.g. RoBERTa, BERT fine-tuned on GoEmotions) to classify emotions expressed in each text.
- Examine how specific emotions (e.g. fear, anger, pride) are used across parties, time periods, or topics to shape opinion.
- Create plots or dashboards comparing emotional tone across actors, media types, or ideological groups.
- Apply explainability methods (e.g. SHAP, attention heatmaps) to highlight emotional trigger words and rhetorical patterns.

## AI Disclosure

AI-generated content has been used in this project.

**Model Used:** Claude 4 , ChatGpt

**Purposes:**
- Generating and improving testing scripts
- Enhancing the fine-tuning process implementation
- Debugging and fixing issues in comparison report generation
- Code optimization and error resolution

**Extent of Integration:**
All AI-generated code was reviewed and modified as needed, and integrated only after verifying that the changes aligned with the project requirements. The project's scope, structure, methodology, evaluation logic, and code were developed independently, with AI assistance for improvements and debugging.

## Project Structure

### 1. Data Collection (`1_data_collection.py`)
Collects tweets from politicians listed in a CSV file using the Twitter API.

**Features:**
- Fetches tweets from January 1, 2021 to present
- Skips retweets and replies to get original content
- Handles rate limiting with automatic retry
- Saves tweets incrementally to prevent data loss
- Supports custom CSV file path as command line argument

**Usage:**
```bash
# Using default politicians.csv file and 100 tweet limit
python 1_data_collection.py

# Using custom CSV file with default 100 tweet limit
python 1_data_collection.py path/to/your/politicians.csv

# Using custom CSV file and custom tweet limit
python 1_data_collection.py path/to/your/politicians.csv 1000
```

### 2. Data Cleaning and Combination (`2_combine_clean_all_tweets.py`)
Combines all collected tweet files and cleans the text data.

**Features:**
- Combines all CSV files from raw data folder
- Removes URLs, mentions, and emojis
- Filters out short tweets and single words
- Merges with politician party information
- Exports cleaned data to combined CSV file
- Supports custom politicians CSV file path as command line argument

**Usage:**
```bash
# Using default politicians.csv file
python 2_combine_clean_all_tweets.py

# Using custom politicians CSV file
python 2_combine_clean_all_tweets.py path/to/your/politicians.csv
```

### 3. BERT Model Fine-tuning (`3_fine_tune_bert_model.py`)
Fine-tunes a Turkish BERT model for emotion classification using the TrEmo dataset.

**Features:**
- Uses Turkish BERT model (dbmdz/bert-base-turkish-cased by default)
- Handles class imbalance with weighted loss
- Early stopping and best model selection
- Comprehensive evaluation with confusion matrix
- Saves fine-tuned model and label mappings
- Supports custom model name as command line argument

**Usage:**
```bash
# Using default Turkish BERT model
python 3_fine_tune_bert_model.py

# Using custom model
python 3_fine_tune_bert_model.py savasy/bert-base-turkish-sentiment-cased
```

**Supported Models:**
- `dbmdz/bert-base-turkish-cased` (default)
- Any compatible  BERT model from Hugging Faces

### 4. Topic Extraction (`4_topic_extractor.py`)
Performs zero-shot topic classification on collected tweets.

**Features:**
- Uses mDeBERTa-v3 (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) for multilingual topic classification
- Batch processing for efficiency
- 15 predefined topic categories
- GPU acceleration support
- Adds topic column to existing dataset
- Supports custom topic labels via command line arguments

**Topics:**
- göç, ekonomi, eğitim, sağlık, adalet, güvenlik
- dış politika, sosyal politikalar, çevre, ulaşım
- enerji, kültür ve medya, siyaset, yerel yönetim, genel

**Usage:**
```bash
# Using default topics
python 4_topic_extractor.py

# Using custom topics (comma-separated)
python 4_topic_extractor.py --topics "göç, ekonomi, eğitim, sağlık, adalet, güvenlik, dış politika, sosyal politikalar, çevre, ulaşım, enerji, kültür ve medya, siyaset, yerel yönetim, genel"
```

### 5. Emotion Classification (`5_emotions_classification_of_tweets.py`)
Applies the fine-tuned emotion model to classify emotions in collected tweets.

**Features:**
- Uses the fine-tuned Turkish BERT model from step 3
- Predicts emotions for all collected tweets
- Returns high-confidence emotions (>5% probability)
- Batch processing for efficiency
- Saves tweets with emotion predictions

**Output:**
- `predicted_emotions`: High-confidence emotions (>5%)
- `top3_emotions`: Top 3 emotion predictions regardless of confidence

**Usage:**
```bash
python 5_emotions_classification_of_tweets.py
```

### 6. Sentiment Analysis and Visualization (`6_sentiment_analyses.py`)
Creates comprehensive visualizations and analysis of emotions across political dimensions.

**Features:**
- Political side emotion comparison (left vs right)
- Party-based emotion distribution heatmaps
- Individual politician emotion profiles
- Overall emotion distribution analysis
- Dashboard-style visualization reports

**Visualizations:**
- Political side distribution
- Overall emotion distribution
- Left vs right emotion comparison
- Party emotion heatmaps
- Individual author emotion radar charts
- Comprehensive dashboard

**Usage:**
```bash
python 6_sentiment_analyses.py
```

## Politicians CSV Format

The `politicians.csv` file contains 28 Turkish politicians from 6 different parties you can change it !!!:

**Column Descriptions:**
- `Author`: Twitter username (without @ symbol)
- `party`: Political party abbreviation (AKP, CHP, MHP, IYI, DEM, ZAFER, TIP)
- `political_side`: Political orientation (left/right)

## Setup Requirements

1. Create a `.env` file with your Twitter credentials:
```
EMAIL=your_email@gmail.com
USERNAME=your_twitter_username
PASSWORD=your_password
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the TrEmo dataset and place `tremo_data.csv` in `2_turkish_emotions_datasets/` folder

## Output Structure

- `0_politican_tweets_raw_data/`: Individual CSV files for each politician
- `1_politican_tweets_combined_data/`: Combined and cleaned tweet data
- `2_turkish_emotions_datasets/`: Emotion classification datasets (TrEmo)
- `3_tweets_with_emotions/`: Tweets with emotion predictions
- `4_sentiment_plots/`: Visualization plots and charts
- `fine_tuned_turkish_emotions/`: Fine-tuned BERT model
- `3_tweets_with_emotions/all_cleaned_tweets_with_topics_and_emotions.csv`: Final processed dataset with topics and emotions

## Workflow

1. **Data Collection**: Run `1_data_collection.py` to gather tweets
2. **Data Cleaning**: Run `2_combine_clean_all_tweets.py` to clean and combine data
3. **Model Training**: Run `3_fine_tune_bert_model.py` to train emotion classifier with tremo_data.csv (includes text and their emotions)
4. **Topic Extraction**: Run `4_topic_extractor.py` to add topic labels
5. **Emotion Classification**: Run `5_emotions_classification_of_tweets.py` to predict emotions
6. **Analysis & Visualization**: Run `6_sentiment_analyses.py` to create comprehensive analysis


## Analysis Features

- **Emotion Classification**: Multi-class emotion detection using fine-tuned Turkish BERT
- **Topic Classification**: Zero-shot topic classification with 15 categories
- **Political Analysis**: Comparison between left/right political orientations
- **Party Comparison**: Emotion distribution across different political parties
- **Individual Profiles**: Detailed emotion analysis for top politicians
- **Visualization**: Comprehensive charts, heatmaps, and radar plots
- **Party Comparison**: Emotion distribution across different political parties
- **Individual Profiles**: Detailed emotion analysis for top politicians
- **Visualization**: Comprehensive charts, heatmaps, and radar plots
