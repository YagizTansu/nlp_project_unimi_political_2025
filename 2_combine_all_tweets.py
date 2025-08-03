import pandas as pd
import os
import glob
import re
from datetime import datetime

def combine_tweet_data():
    """
    Combine all CSV files from raw_data folder and merge with politician parties data
    """
    # Read politician parties data
    parties_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/politician_parties.csv')

    # Get all CSV files from raw_data folder
    raw_data_path = '/home/yagiz/Desktop/nlp_project/politican_tweets_raw_data/*.csv'
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
    
    # Reorder columns for better readability
    columns_order = ['ID', 'Author', 'party', 'political_side', 'Text', 'Date', 
                    'Like_Count', 'Retweet_Count', 'Reply_Count', 'URL']
    full_tweets = full_tweets[columns_order]
    
    # Save the result
    output_file = '/home/yagiz/Desktop/nlp_project/politican_tweets_combined_data/full_tweets.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    full_tweets.to_csv(output_file, index=False)
    
    print(f"Successfully combined {len(csv_files)} CSV files")
    print(f"Total tweets: {len(full_tweets)}")
    
    return full_tweets

def preprocess_tweets(df=None, input_file=None, output_file=None):
    """
    Preprocess tweets data by selecting required columns and cleaning text
    
    Parameters:
    df: DataFrame (optional) - if provided, will process this dataframe
    input_file: str (optional) - path to input CSV file
    output_file: str (optional) - path to save preprocessed data
    """
    # Load data if not provided
    if df is None:
        if input_file is None:
            input_file = '/home/yagiz/Desktop/nlp_project/full_tweets.csv'
        df = pd.read_csv(input_file)
    
    # Select required columns
    required_columns = ['ID', 'Author', 'party', 'political_side', 'Text', 'Date']
    preprocessed_df = df[required_columns].copy()
    
    # Remove rows with missing text
    preprocessed_df = preprocessed_df.dropna(subset=['Text'])
    
    # Clean text data
    preprocessed_df['Text'] = preprocessed_df['Text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    preprocessed_df = preprocessed_df[preprocessed_df['Text'].str.strip() != '']
    
    # Save preprocessed data
    if output_file is None:
        output_file = '/home/yagiz/Desktop/nlp_project/politican_tweets_combined_data/preprocessed_tweets.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    preprocessed_df.to_csv(output_file, index=False)
    
    print(f"Preprocessing completed!")
    print(f"Original tweets: {len(df)}")
    print(f"Preprocessed tweets: {len(preprocessed_df)}")
    print(f"Saved to: {output_file}")
    
    return preprocessed_df

def clean_text(text):
    """
    Minimal cleaning - only remove URLs and extra whitespace
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Only remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def get_data_statistics(df=None, input_file=None):
    """
    Get basic statistics about the preprocessed data
    """
    if df is None:
        if input_file is None:
            input_file = '/home/yagiz/Desktop/nlp_project/preprocessed_tweets.csv'
        df = pd.read_csv(input_file)
    
    print("=== Data Statistics ===")
    print(f"Total tweets: {len(df)}")
    print(f"Unique authors: {df['Author'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("\n=== Tweets by Party ===")
    print(df['party'].value_counts())
    
    print("\n=== Tweets by Political Side ===")
    print(df['political_side'].value_counts())
    
    print("\n=== Tweets by Author ===")
    print(df['Author'].value_counts())
    
    return df

if __name__ == "__main__":
    # Step 1: Combine all CSV files
    print("Step 1: Combining CSV files...")
    full_tweets_df = combine_tweet_data()
    
    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing tweets...")
    preprocessed_df = preprocess_tweets(df=full_tweets_df)
    
    # Step 3: Get statistics
    print("\nStep 3: Data statistics...")
    get_data_statistics(df=preprocessed_df)
