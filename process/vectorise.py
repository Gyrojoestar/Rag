from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.corpus import gutenberg

#import austen emma as text
emma = gutenberg.raw("austen-emma.txt")

#initialise the vectorizer
vectorizer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# chunker using langchain
chunker = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)

texts = chunker.split_text(emma)
embeddings = vectorizer.encode(texts)

print(f"Number of chunks: {len(texts)}\n{texts[:2]}")
print(f"Embedding shape: {embeddings.shape}")