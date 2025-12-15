import torch
from transformers import AutoModel, AutoTokenizer


class CodeEmbedder:
    """Singleton do obsługi modelu CodeBERT/RoBERTa"""

    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text):
        """Zwraca wektor (768,) reprezentujący kod/tekst"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Używamy CLS token embedding (pierwszy token) jako reprezentacji całości
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    def get_token_embeddings(self, text):
        """Zwraca macierz (seq_len, 768)"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.cpu().numpy()[0]


embedder = CodeEmbedder()
