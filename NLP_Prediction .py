
import numpy as np
import nltk as nl
import subprocess
nl.download('punkt')
subprocess.run("pip install nltk", shell=True)

from nltk.tokenize import word_tokenize


def tokenizeinput(text_input):
    return word_tokenize(text_input)

print("Tokenized input: ", tokenizeinput("This is a test."))
