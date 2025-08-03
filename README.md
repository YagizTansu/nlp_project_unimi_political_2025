# CSV Translation Tool

This tool translates the emotion dataset from English to Turkish.

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the translation script:
```bash
python translate_csv.py
```

2. The translated file will be saved as `train_turkish.csv` in the same directory.

## Features

- Preserves the original CSV structure (text, label, id)
- Handles translation errors gracefully
- Shows progress during translation
- Adds delays to avoid rate limiting

## Output

The output file will have the same format as the input:
- Column 1: Translated Turkish text
- Column 2: Original emotion labels
- Column 3: Original IDs
