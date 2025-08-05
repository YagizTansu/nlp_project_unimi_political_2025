import pandas as pd
import os

def convert_emotion_labels():
    # Read the original dataset
    df = pd.read_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/Emotion_dataset_train.csv', index_col=0)
    
    # Get unique emotions and sort them for consistency
    unique_emotions = sorted(df['Label'].unique())
    
    # Create emotion to number mapping
    emotion_to_num = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    
    # Create the emotions mapping CSV
    emotions_df = pd.DataFrame({
        'emotion_id': list(emotion_to_num.values()),
        'emotion_name': list(emotion_to_num.keys())
    })
    
    # Save emotions mapping
    emotions_df.to_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/emotions.csv', index=False)
    
    # Convert labels to numerical values
    df['Label_Numeric'] = df['Label'].map(emotion_to_num)
    
    # Create new dataset with numerical labels
    df_numeric = df[['Sentence', 'Label_Numeric']].copy()
    df_numeric.columns = ['Sentence', 'Label']
    
    # Save the new dataset
    df_numeric.to_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/Emotion_dataset_train_numeric.csv', index=True)
    
    print("Conversion completed!")
    print(f"Found {len(unique_emotions)} unique emotions:")
    for emotion, num in emotion_to_num.items():
        print(f"  {num}: {emotion}")
    
    print(f"\nFiles created:")
    print(f"  - emotions.csv (emotion mapping)")
    print(f"  - Emotion_dataset_train_numeric.csv (dataset with numerical labels)")

if __name__ == "__main__":
    convert_emotion_labels()
