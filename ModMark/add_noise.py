import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import random
import string

def add_noise(word, seed=42, p_r=0.1, p_i=0.01, p_d=0.01):
    random.seed(seed)
    def replace_char(word):
        new_word = list(word)
        for i in range(len(new_word)):
            if random.random() < p_r:
                new_word[i] = random.choice(string.ascii_letters)
        return ''.join(new_word)

    def insert_char(word):
        if random.random() < p_i:
            index = random.randint(0, len(word))
            random_char = random.choice(string.ascii_letters)
            word = word[:index] + random_char + word[index:]
        return word

    def delete_char(word):
        if random.random() < p_d and len(word) > 0:
            index = random.randint(0, len(word) - 1)
            word = word[:index] + word[index + 1:]
        return word

    word = replace_char(word)
    word = insert_char(word)
    word = delete_char(word)
    return word

words = ["watch", "calculate"]

for word in words:
    word = add_noise(word)
    print(word, "\n")