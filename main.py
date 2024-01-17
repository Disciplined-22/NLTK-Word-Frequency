import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Sample unstructured text data
text_data = """
Natural language processing is a field of artificial intelligence that focuses on the interaction between computers and humans using natural language. It involves the development of algorithms and models to enable computers to understand, interpret, and generate human-like text.
NLTK, or Natural Language Toolkit, is a powerful library for working with human language data in Python. It provides tools for tasks such as tokenization, stemming, part-of-speech tagging, and more.
Text mining is the process of extracting valuable information and knowledge from unstructured text data. It includes tasks like sentiment analysis, topic modeling, and document clustering.
"""

# Tokenization
tokens = word_tokenize(text_data.lower())  # Convert to lowercase for consistency

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Frequency Distribution
fdist = FreqDist(filtered_tokens)

# Display the most common words
print("Most common words:")
for word, frequency in fdist.most_common(5):
    print(f"{word}: {frequency}")
