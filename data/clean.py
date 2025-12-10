import re
from bs4 import BeautifulSoup
from tqdm import tqdm

# Define input and output file paths
input_path = "c4.txt"
output_path = "c4_cleaned.txt"

# Regular expressions for removing URLs and email addresses
url_pattern = re.compile(r'https?://\S+|www\.\S+')
email_pattern = re.compile(r'\S+@\S+')
whitespace_pattern = re.compile(r'\s+')

def clean_text(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove URLs
    text = url_pattern.sub('', text)

    # Remove email addresses
    text = email_pattern.sub('', text)

    # Normalize whitespace
    text = whitespace_pattern.sub(' ', text)

    # Strip leading and trailing whitespace
    return text.strip()

# Process the input file and write cleaned text to the output file
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Cleaning"):
        cleaned_line = clean_text(line)
        if cleaned_line:  # Write non-empty lines
            fout.write(cleaned_line + "\n")

print(f"Cleaning complete. Cleaned data saved to {output_path}")
