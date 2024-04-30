import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load and preprocess the data
f = open("./data/dictionary.txt", "r")
dictionary_text = f.read()
f.close()


f = open("./data/text.txt", "r")
text_text = f.read()
f.close()


f = open("./data/gutenberg.txt", "r")
gutenberg_text = f.read()
f.close()

f = open("./data/gutenberg2.txt", "r")
gutenberg_text += f.read()

f = open("./data/gutenberg3.txt", "r")
gutenberg_text += f.read()

f = open("./data/gutenberg4.txt", "r")
gutenberg_text += f.read()




dictionary_text += text_text + gutenberg_text

# Tokenize the text
dictionary_tokens = word_tokenize(dictionary_text)

# Reduce vocabulary size by considering only the most common words
vocab_size = 150000 # Adjust as needed
word_freq = nltk.FreqDist(dictionary_tokens)
top_words = [word for word, _ in word_freq.most_common(vocab_size)]
training_data = [word if word in top_words else "<UNK>" for word in dictionary_tokens]

# Train Word2Vec model
model = Word2Vec(sentences=[training_data], vector_size=100, window=5, min_count=1, workers=4, sg=1)  # Use Skipgram (sg=1)

# Define a function to generate context-target pairs
def generate_context_target_pairs(tokens, window_size=2):
    pairs = []
    for i, target_word in enumerate(tokens):
        if(i % 1000 == 0):
            print("Processing token", i, "out of", len(tokens))
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                context_word = tokens[j]
                pairs.append((context_word, target_word))
    return pairs

# Generate context-target pairs for evaluation
evaluation_pairs = generate_context_target_pairs(training_data)

# Evaluate the model by predicting context words for target words
random_word = "The"

# Initialize sequence with the random word
sequence = [random_word]
for i in range(19):
    # Predict the next word based on the last word in the sequence
    last_word = sequence[-1]
    predicted_contexts = [predicted[0] for predicted in model.wv.most_similar(last_word)]
    next_word = np.random.choice(predicted_contexts)
    # Append the predicted word to the sequence
    sequence.append(next_word)

# Print the generated sequence
print("Generated Sequence:")
print(" ".join(sequence))

random_word = "A"

# Initialize sequence with the random word
sequence = [random_word]
for i in range(19):
    # Predict the next word based on the last word in the sequence
    last_word = sequence[-1]
    predicted_contexts = [predicted[0] for predicted in model.wv.most_similar(last_word)]
    next_word = np.random.choice(predicted_contexts)
    # Append the predicted word to the sequence
    sequence.append(next_word)

# Print the generated sequence
print("Generated Sequence:")
print(" ".join(sequence))

random_word = "Bible"

# Initialize sequence with the random word
sequence = [random_word]
for i in range(3000):
    # Predict the next word based on the last word in the sequence
    last_word = sequence[-1]
    predicted_contexts = [predicted[0] for predicted in model.wv.most_similar(last_word)]
    next_word = np.random.choice(predicted_contexts)
    # Append the predicted word to the sequence
    sequence.append(next_word)

# Print the generated sequence
print("Generated Sequence:")
print(" ".join(sequence))