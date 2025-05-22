import os
from utils.utils import read_input_str

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_text = read_input_str(os.path.join(BASE_DIR, "input_txt", "input.txt")) 

print(input_text)