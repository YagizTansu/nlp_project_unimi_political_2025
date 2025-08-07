import asyncio
from datetime import datetime
import pandas as pd
from twikit import Client
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

USERNAME = os.getenv("USERNAME")
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

# Initialize client
client = Client('en-US')

def parse_twitter_date(date_str):
    """Convert Twitter date format to datetime object"""
    return datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')

def save_tweets_to_csv(tweets, username, filename=None, append=False):
    """Save tweets to CSV file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_tweets_{timestamp}.csv"
    
    # Prepare tweet data as list
    tweet_data = []
    for tweet in tweets:
        # Get full text if available, otherwise use regular text
        full_text = getattr(tweet, 'full_text', tweet.text)
        
        tweet_data.append({
            'ID': tweet.id,
            'Author': tweet.user.screen_name,
            'Text': full_text,  # Use full text instead of truncated text
            'Date': tweet.created_at
            # 'Like_Count': tweet.favorite_count,
            # 'Retweet_Count': tweet.retweet_count,
            # 'Reply_Count': getattr(tweet, 'reply_count', 0),
            # 'URL': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(tweet_data)
    
    # Check if file exists and we want to append
    if append and os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"Appended {len(tweets)} tweets to '{filename}'")
    else:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Tweets saved to '{filename}' file.")
    
    return filename

async def get_user_tweets(username, limit=10):
    # Find user
    user = await client.get_user_by_screen_name(username)
    if not user:
        print(f"User '{username}' not found.")
        return [], None

    print(f"Fetching tweets from '{username}' user...")

    # Set January 1, 2021 date
    start_date = datetime(2021, 1, 1, tzinfo=datetime.now().astimezone().tzinfo)
    
    all_tweets = []
    tweets = await user.get_tweets(count=10, tweet_type='Tweets')
    
    # Create initial filename for progress saves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create raw_data directory if it doesn't exist
    raw_data_dir = "0_politican_tweets_raw_data"
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    
    csv_filename = f"{raw_data_dir}/{username}_tweets_{timestamp}.csv"
    
    
    while tweets and len(all_tweets) < limit:
        batch_tweets = []
        for tweet in tweets:
            # Check tweet date
            tweet_date = parse_twitter_date(tweet.created_at)
            
            # If tweet is older than January 1, 2024, break the loop
            if tweet_date < start_date:
                print(f"Found tweet older than January 1, 2024. Stopping search.")
                tweets = None  # End the loop
                break
            
            # Skip retweets
            if tweet.text.startswith("RT @"):
                continue  # Retweet ise atla
            
            # Skip replies
            if hasattr(tweet, 'in_reply_to_status_id') and tweet.in_reply_to_status_id is not None:
                continue  # Cevap (reply) ise atla
            
            batch_tweets.append(tweet)
            if len(all_tweets) + len(batch_tweets) >= limit:
                break
        
        # Add batch to all tweets
        all_tweets.extend(batch_tweets)
        
        # Save tweets immediately after each batch
        if batch_tweets:  # Only save if there are tweets in the batch
            append_mode = len(all_tweets) > len(batch_tweets)
            save_tweets_to_csv(batch_tweets, username, csv_filename, append=append_mode)
            print(f"Progress: {len(all_tweets)}/{limit} tweets saved")
        
        if not tweets or len(all_tweets) >= limit:
            break
            
        print(f"Fetched {len(all_tweets)} tweets so far...")
        
        # Get more tweets through pagination with improved retry logic
        next_tweets = None
        while True:  # Keep trying until we get tweets or face non-rate-limit error
            try:
                print(f"Attempting to get next page of tweets...")
                next_tweets = await tweets.next()
                print(f"Successfully fetched next page")
                break
                
            except Exception as e:
                print(f"Error fetching more tweets: {e}")
                
                if "429" in str(e) or "Rate limit exceeded" in str(e):
                    wait_time = 950  # 16 minutes  in seconds
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds (16 minutes and 15 seconds) before retry...")

                    # Countdown timer
                    for remaining in range(wait_time, 0, -1):
                        minutes = remaining // 60
                        seconds = remaining % 60
                        print(f"\rRemaining time: {minutes}:{seconds:02d}", end='', flush=True)
                        await asyncio.sleep(1)
                    
                    print("\nRetrying...")
                    continue  # Try again after waiting
                else:
                    # For other errors, break and stop trying
                    print(f"Non-rate-limit error encountered. Stopping pagination.")
                    break
        
        # Check if we got new tweets
        if next_tweets:
            tweets = next_tweets
        else:
            print(f"Failed to get next page, continuing to next iteration...")
            continue  # Continue to next iteration instead of breaking
    
    # Take only the requested number of tweets
    all_tweets = all_tweets[:limit]

    if all_tweets:
        print(f"Total {len(all_tweets)} tweets fetched (January 1, 2024 - present).")
        
        # Print to console (optional) - limit to first 5 tweets to avoid spam
        for i, tweet in enumerate(all_tweets[:5]):
            print(f"--- Tweet {i+1} ---")
            print(f"Author: @{tweet.user.screen_name}")
            print(f"ID: {tweet.id}")
            print(f"Text: {tweet.text[:150]}...") # Show first 150 characters of tweet
            print(f"Date: {tweet.created_at}")
            print("-" * 20)
        
        return all_tweets, csv_filename
    else:
        print(f"No tweets found or could not fetch tweets from '{username}' user after January 1, 2024.")
        return [], None

async def process_all_politicians(csv_file_path, tweet_limit=200):
    """Process all politicians from the CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Found {len(df)} politicians in the CSV file")
        
        # Process each politician
        for index, row in df.iterrows():
            username = row['Author']
            party = row['party']
            political_side = row['political_side']
            
            # Remove @ symbol if present
            clean_username = username.replace('@', '')
            
            print(f"\n{'='*50}")
            print(f"Processing politician {index + 1}/{len(df)}")
            print(f"Author: {username}")
            print(f"Party: {party}")
            print(f"Political Side: {political_side}")
            print(f"{'='*50}")
            
            try:
                tweets, csv_file = await get_user_tweets(clean_username, tweet_limit)
                if csv_file:
                    print(f"✓ Successfully saved {len(tweets)} tweets for {username}")
                else:
                    print(f"✗ Failed to collect tweets for {username}")
            except Exception as e:
                print(f"✗ Error processing {username}: {e}")
                continue
            
            # Add a small delay between users to be respectful to the API
            print(f"Waiting 10 seconds before processing next politician...")
            await asyncio.sleep(10)
        
        print(f"\n{'='*50}")
        print("All politicians have been processed!")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")

async def main():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
        cookies_file='cookies.json'
    )
    
    # Get CSV file path from command line argument or use default
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        print(f"Using CSV file: {csv_file_path}")
    else:
        csv_file_path = "politicians.csv"
        print(f"Using default CSV file: {csv_file_path}")
    
    # Get tweet limit from command line argument or use default
    if len(sys.argv) > 2:
        try:
            tweet_limit = int(sys.argv[2])
            print(f"Using tweet limit: {tweet_limit}")
        except ValueError:
            print(f"Invalid tweet limit '{sys.argv[2]}', using default: 100")
            tweet_limit = 100
    else:
        tweet_limit = 100
        print(f"Using default tweet limit: {tweet_limit}")
    
    # Process all politicians from CSV
    await process_all_politicians(csv_file_path, tweet_limit=tweet_limit)

asyncio.run(main())