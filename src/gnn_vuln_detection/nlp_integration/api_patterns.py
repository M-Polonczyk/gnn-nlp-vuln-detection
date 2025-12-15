import re

import numpy as np


class APIPatternLearner:
    def __init__(self, embedder):
        self.embedder = embedder
        # Definicje ryzykownych API i słów kluczowych sugerujących walidację
        self.unsafe_apis = ["strcpy", "gets", "sprintf", "malloc", "free"]
        self.validation_keywords = ["if", "len", "sizeof", "check", "assert", "return"]

    def extract_api_context(self, source_code, window_size=2):
        """
        Znajduje niebezpieczne API i pobiera ich kontekst (linie przed/po).
        """
        lines = source_code.split("\n")
        risks = []

        for i, line in enumerate(lines):
            for api in self.unsafe_apis:
                # Prosty regex wykrywający wywołanie funkcji (np. strcpy(...))
                if re.search(rf"\b{api}\s*\(", line):
                    # Pobierz okno kontekstowe
                    start = max(0, i - window_size)
                    end = min(len(lines), i + window_size + 1)
                    context_snippet = "\n".join(lines[start:end])

                    # Analiza heurystyczna w oknie
                    has_validation = any(
                        vk in context_snippet for vk in self.validation_keywords
                    )

                    # Oblicz embedding kontekstu
                    context_emb = self.embedder.get_embedding(context_snippet)

                    risks.append(
                        {
                            "api": api,
                            "line_no": i + 1,
                            "context_embedding": context_emb,  # Vector (768,)
                            "has_validation_hint": has_validation,
                            "risk_score": 0.2 if has_validation else 0.9,
                        }
                    )
        return risks

    def get_aggregated_risk(self, risks):
        """Zwraca uśredniony wektor ryzyka dla całej funkcji"""
        if not risks:
            return 0.0, np.zeros(768)

        avg_risk = np.mean([r["risk_score"] for r in risks])
        # Średnia ważona embeddingów (można użyć Attention w wersji zaawansowanej)
        avg_emb = np.mean([r["context_embedding"] for r in risks], axis=0)

        return avg_risk, avg_emb
