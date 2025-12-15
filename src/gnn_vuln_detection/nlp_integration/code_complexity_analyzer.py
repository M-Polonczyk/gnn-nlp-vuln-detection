import re

import numpy as np


class CodeComplexityAnalyzer:
    def __init__(self, embedder):
        self.embedder = embedder

    def compute_semantic_variance(self, source_code):
        """
        Oblicza odchylenie standardowe embeddingów tokenów.
        Wysoka wariancja = kod wykonuje bardzo różnorodne operacje (cognitive load).
        """
        # Pobierz embeddingi dla każdego tokenu (macierz N x 768)
        token_embeddings = self.embedder.get_token_embeddings(source_code)

        # Oblicz std dev wzdłuż wymiaru sekwencji, potem uśrednij
        # Wariancja "tematyczna" kodu
        variance = np.std(token_embeddings, axis=0).mean()
        return variance

    def compute_cyclomatic_complexity(self, source_code):
        """
        Prosta aproksymacja na podstawie słów kluczowych sterowania.
        CC = Edges - Nodes + 2 ~= Branches + 1
        """
        branches = ["if", "else", "while", "for", "case", "&&", "||", "catch"]
        count = 0
        tokens = re.findall(r"\w+|&&|\|\|", source_code)
        for t in tokens:
            if t in branches:
                count += 1
        return count + 1

    def get_complexity_features(self, source_code):
        return {
            "cyclomatic": self.compute_cyclomatic_complexity(source_code),
            "semantic_variance": float(self.compute_semantic_variance(source_code)),
            "loc": len(source_code.split("\n")),
        }
