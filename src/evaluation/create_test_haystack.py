import os
import random

def create_test_haystack(haystack_path="./test_haystack"):
    
    if os.path.isabs(haystack_path):
        base_dir = ""
    else:
        base_dir = os.getcwd()
    haystack_dir = os.path.abspath(os.path.join(base_dir, haystack_path))
    
    # Create the test haystack directory
    os.makedirs(haystack_dir, exist_ok=True)

    # Sample text to use as the base content
    base_texts = [
        "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the English alphabet.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
        "Python is a high-level, interpreted programming language known for its readability and versatility across various domains.",
        "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data.",
        "Transformers are deep learning models introduced in 2017 that have revolutionized natural language processing tasks."
    ]

    # Create 10 sample documents
    for i in range(10):
        document_content = []
        
        # Add 5-10 paragraphs to each document
        num_paragraphs = random.randint(5, 10)
        for _ in range(num_paragraphs):
            # Select a random base text
            text = random.choice(base_texts)
            document_content.append(text)
        
        # Add the needle to document 4 and document 8 at different positions
        if i == 3:
            pos = random.randint(1, len(document_content)-1)
            document_content.insert(pos, "The brown fox jumped over the lazy dog. This is our needle hidden in the text.")
        elif i == 7:
            pos = random.randint(1, len(document_content)-1)
            document_content.insert(pos, "The brown fox jumped over the lazy dog. This is another instance of our needle.")

        # Write the document to a file
        doc_path = os.path.join(haystack_dir, f"document_{i+1}.txt")
        with open(doc_path, "w") as f:
            f.write("\n\n".join(document_content))

    print("Created test_haystack folder with 10 documents. The needle 'The brown fox jumped over the lazy dog.' is hidden in documents 4 and 8.")

if __name__ == "__main__":
    create_test_haystack()