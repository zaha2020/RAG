import fitz
import re
import torch
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class PDFProcessor:
    def __init__(self, pdf_path, model_embedd, tokenizer_embedd, model_name, tokenizer_name):
        self.pdf_path = pdf_path
        self.tokenizer_embedd = AutoTokenizer.from_pretrained(tokenizer_embedd)
        self.model_embedd = AutoModel.from_pretrained(model_embedd)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.text_chunks = []
        self.index = None

    def extract_text_from_pdf(self):
        text = ""
        doc = fitz.open(self.pdf_path)
        for page in doc:
            text += page.get_text()
        return text

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text_by_sentence(self, text, max_chunk_size=500, overlap_size=50):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(' '.join(current_chunk))
                overlap_text = ' '.join(current_chunk)[-overlap_size:]
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def embed_text(self, text):
        inputs = self.tokenizer_embedd(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model_embedd(**inputs).last_hidden_state.mean(dim=1)
        return embeddings

    def build_faiss_index(self):
        chunk_embeddings = []
        for chunk in self.text_chunks:
            chunk_embedding = self.embed_text(chunk)
            chunk_embeddings.append(chunk_embedding)

        embedding_matrix = torch.vstack(chunk_embeddings).numpy()
        self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        self.index.add(embedding_matrix)

    def process_pdf(self):
        raw_text = self.extract_text_from_pdf()
        cleaned_text = self.clean_text(raw_text)
        self.text_chunks = self.chunk_text_by_sentence(cleaned_text)
        self.build_faiss_index()

    def query_database(self, query_text, top_k=1):
        query_embedding = self.embed_text(query_text).numpy()
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0]

    def generate_response(self, retrieved_chunks):
        input_text = " ".join(retrieved_chunks)
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=5,
                repetition_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def retrieve_and_generate_response(self, query_text):
        retrieved_indices = self.query_database(query_text)
        retrieved_chunks = [self.text_chunks[i] for i in retrieved_indices]
        response = self.generate_response(retrieved_chunks)
        return response
