import gzip
from datasets import load_dataset
from tqdm import tqdm

# Define the output file path
output_path = "c4_subset_3gb.txt.gz"

# Set the maximum number of bytes (3 GB)
max_bytes = 3 * 1024 * 1024 * 1024  # 3 GB in bytes
total_bytes = 0

# Load the English subset of the C4 dataset in streaming mode
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Open the output file in write-text mode with gzip compression
with gzip.open(output_path, "wt", encoding="utf-8") as f_out:
    for example in tqdm(dataset, desc="Extracting"):
        text = example.get("text", "").strip()
        if not text:
            continue
        # Write the text followed by two newlines to separate entries
        f_out.write(text + "\n\n")
        # Update the total bytes written
        total_bytes += len(text.encode("utf-8")) + 2  # Account for the added newlines
        # Break the loop if the size limit is reached
        if total_bytes >= max_bytes:
            break

print(f"Extraction complete. Saved to {output_path}")
