import os
from dotenv import load_dotenv
from huggingface_hub import login
import yaml
from pdf_processor import PDFProcessor

load_dotenv()

with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

file_path = config.get('FILE_PATH', '')
model_embedd = config.get('MODEL_EMBEDD', '')
tokenizer_embedd = config.get('TOKENIZER_EMBEDD', '')
model_name = config.get('MODEL_NAME', '')
tokenizer_name = config.get('TOKENIZER_NAME', '')

token = os.getenv("TOKEN")
login(token=token)

processor = PDFProcessor(file_path, model_embedd, tokenizer_embedd, model_name, tokenizer_name)
processor.process_pdf()

query = "چیست؟ Artificial Intelligence لطفاً توضیح دهید که هوش مصنوعی یا"
response = processor.retrieve_and_generate_response(query)
print("Generated Response:", response)