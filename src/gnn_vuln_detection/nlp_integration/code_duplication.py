import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCloneDetector:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vulnerability_db = []  # Lista krotek (embedding, cwe_label)

    def load_known_vulnerabilities(self, vulnerability_samples):
        """
        Ładuje bazę znanych podatnych fragmentów kodu.
        vulnerability_samples: list of strings (source code)
        """
        print(f"Indexing {len(vulnerability_samples)} known vulnerabilities...")
        for code in vulnerability_samples:
            emb = self.embedder.get_embedding(code)
            self.vulnerability_db.append(emb)
        self.vulnerability_db = np.array(self.vulnerability_db)

    def detect_clones(self, target_code, threshold=0.85):
        """
        Sprawdza podobieństwo target_code do bazy podatności.
        """
        if len(self.vulnerability_db) == 0:
            return 0.0, False

        target_emb = self.embedder.get_embedding(target_code).reshape(1, -1)

        # Oblicz podobieństwo cosinusowe ze wszystkimi wzorcami w bazie
        similarities = cosine_similarity(target_emb, self.vulnerability_db)[0]

        max_similarity = np.max(similarities)
        is_clone = max_similarity > threshold

        return max_similarity, is_clone
