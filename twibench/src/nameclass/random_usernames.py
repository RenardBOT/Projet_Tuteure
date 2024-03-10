import random
import string

def generate_random_name(length=15, letter_ratio=0.8):
    letters_count = int(length * letter_ratio)
    digits_count = length - letters_count
    letters = random.choices(string.ascii_letters, k=letters_count)
    digits = random.choices(string.digits, k=digits_count)
    name = ''.join(random.sample(letters + digits, length))
    return name

