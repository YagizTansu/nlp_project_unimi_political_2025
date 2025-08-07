import pandas as pd
import os
import glob
import re
import sys

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove RT (retweet indicators) and common retweet patterns
    text = re.sub(r'\bRT\b\s*@[A-Za-z0-9_]+:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRT\b\s*@[A-ZazlA-Z0-9_]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Remove URLs (including t.co links and other patterns)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Specifically target t.co URLs
    text = re.sub(r'https://t\.co/[a-zA-Z0-9]+', '', text)
    text = re.sub(r'http://t\.co/[a-zA-Z0-9]+', '', text)
    text = re.sub(r't\.co/[a-zA-Z0-9]+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+:?', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove hashtags but keep the text (convert #example to example)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+"
    )
    text = emoji_pattern.sub('', text)
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove special characters but keep Turkish characters and basic punctuation
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?-]', '', text)
    
    # Remove standalone words that are too short (1-2 characters) and common artifacts
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2 or word.lower() in ['ve', 'ya', 'da', 'ki', 'mi', 'mu', 'mü']]
    text = ' '.join(filtered_words)
    
    return text.strip()

def is_single_word_tweet(text):
    if pd.isna(text):
        return True
    
    if not text.strip():
        return True
    
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text)

    # If less than 3 meaningful words, consider it single word
    if len(words) < 3:
        return True
    
    return False

def combine_tweet_data(politicians_csv_path='/home/yagiz/Desktop/nlp_project/politicians.csv'):
    # Read politician parties data
    parties_df = pd.read_csv(politicians_csv_path)

    # Get all CSV files from raw_data folder
    raw_data_path = '/home/yagiz/Desktop/nlp_project/0_politican_tweets_raw_data/*.csv'
    csv_files = glob.glob(raw_data_path)
    
    # Read and combine all CSV files
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Combine all dataframes
    combined_tweets = pd.concat(dataframes, ignore_index=True)
    
    # Merge with politician parties data
    full_tweets = pd.merge(combined_tweets, parties_df, on='Author', how='left')
    
    # Clean the tweet text
    print("Cleaning tweet text...")
    full_tweets['Text'] = full_tweets['Text'].apply(clean_text)
    
    # Filter out single word or too short tweets
    print("Filtering out single word and too short tweets...")
    initial_count = len(full_tweets)
    full_tweets = full_tweets[~full_tweets['Text'].apply(is_single_word_tweet)]
    filtered_count = len(full_tweets)
    print(f"Removed {initial_count - filtered_count} single word/short tweets")
    
    # Reorder columns for better readability
    columns_order = ['ID', 'Author', 'party', 'political_side', 'Text', 'Date']
    full_tweets = full_tweets[columns_order]
    
    # Save the result
    output_file = '/home/yagiz/Desktop/nlp_project/1_politican_tweets_combined_data/all_cleaned_tweets.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    full_tweets.to_csv(output_file, index=False)
    
    print(f"Successfully combined {len(csv_files)} CSV files")
    print(f"Total tweets: {len(full_tweets)}")
    
    return full_tweets


if __name__ == "__main__":
    # Get politicians.csv path from command line argument or use default
    if len(sys.argv) > 1:
        politicians_csv_path = sys.argv[1]
        print(f"Using politicians CSV file: {politicians_csv_path}")
    else:
        politicians_csv_path = '/home/yagiz/Desktop/nlp_project/politicians.csv'
        print(f"Using default politicians CSV file: {politicians_csv_path}")
    
    # Check if the file exists
    if not os.path.exists(politicians_csv_path):
        print(f"Error: Politicians CSV file not found at {politicians_csv_path}")
        sys.exit(1)
    
    # Step 1: Combine all CSV files
    print("Step 1: Combining CSV files...")
    full_tweets_df = combine_tweet_data(politicians_csv_path)