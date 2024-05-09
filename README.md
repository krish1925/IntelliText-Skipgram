# NLTK Word2Vec Experiments

This repository contains experiments utilizing the Natural Language Toolkit (NLTK) for reading multiple books stored within the data folder. It aims to train a Word2Vec model using a user-specified vocabulary size, selecting the most popular words. The trained model facilitates word prediction based on a provided input word using a skip-gram implementation.

## Usage

1. Ensure you have Python installed on your system.
2. Install the required dependencies by running:
  ```bash
   pip install nltk
 ```
4. Navigate to the repository directory.
5. Execute the following command to train the Word2Vec model:
  ```bash
   python train_word2vec.py --vocab_size <vocab_size>
 ```
Replace `<vocab_size>` with your desired vocabulary size.

5. After training, run the following command to predict the next word:
  ```bash
python predict_next_word.py --input_word <input_word>
 ```
Replace `<input_word>` with the word for which you want to predict the next word.

## Additional Skip-Gram Implementation

In addition to the NLTK-based Word2Vec implementation, this repository includes another skip-gram method. It allows you to specify a library of your choice (default: NumPy) to extract Python code snippets from `.py` files. These snippets are then utilized to train a skip-gram model using NLTK.

### Usage

1. Follow steps 1-3 from the NLTK Word2Vec Usage section.
2. Execute the following command to train the skip-gram model using code snippets:

Replace `<library_name>` with the desired library (e.g., numpy).

## Important Links

- [NLTK Documentation](https://www.nltk.org/)
- [NLTK Word2Vec Documentation](https://www.nltk.org/howto/word2vec.html)
- [NumPy GitHub Repository](https://github.com/numpy/numpy)

Feel free to explore and experiment with different parameters and functionalities!
