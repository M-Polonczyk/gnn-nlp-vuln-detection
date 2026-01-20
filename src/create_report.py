import csv
import glob
import json
import os

# --- KONFIGURACJA ---
REPORT_CSV_FILE = "docs/corgea_report.csv"  # Nazwa Twojego pliku z raportem CSV
DIVERSEVUL_FILE = (
    "data/preprocessed/diversevul/diversevul.json"  # Nazwa pliku JSON z datasetu
)
SOURCE_DIR = "data/code/diversevul"  # Katalog z plikami .c (kropka to bieżący)
OUTPUT_FILE = "output.json"  # Plik wynikowy


def load_diversevul_data(filepath):
    """
    Wczytuje diversevul.json i tworzy mapę: hash_numeryczny -> {cwe, project}.
    Obsługuje format listy JSON oraz format JSON-Lines (jeden obiekt na linię).
    """
    lookup = {}

    if not os.path.exists(filepath):
        print(
            f"Uwaga: Nie znaleziono pliku {filepath}. Pola 'cwes' i 'project' mogą być puste."
        )
        return lookup

    with open(filepath, encoding="utf-8") as f:
        # Próba wczytania jako standardowy plik JSON
        try:
            content = json.load(f)
            # Jeśli to pojedynczy obiekt, zamknij go w liście
            if isinstance(content, dict):
                content = [content]
        except json.JSONDecodeError:
            # Fallback: Próba wczytania linia po linii (JSON Lines)
            f.seek(0)
            content = []
            for line in f:
                if line.strip():
                    try:
                        content.append(json.loads(line))
                    except:
                        pass

    for entry in content:
        # Kluczem jest "hash" z diversevul (to ten długi numer na końcu nazwy pliku)
        # Konwertujemy na string, aby pasował do nazwy pliku
        key = str(entry.get("hash", ""))
        lookup[key] = {
            "cwes": entry.get("cwe", []),
            "project": entry.get("project", "unknown"),
        }

    return lookup


def load_corgea_findings(filepath):
    """
    Wczytuje raport CSV i grupuje znaleziska po nazwie pliku.
    Zwraca: filename -> lista znalezionych CWE
    """
    findings = {}

    if not os.path.exists(filepath):
        print(f"Błąd: Nie znaleziono pliku raportu {filepath}")
        return findings

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
            filename = row.get("File")
            cwe = row.get("Classification ID")

            if filename:
                if filename not in findings:
                    findings[filename] = set()
                if cwe:
                    findings[filename].add(cwe)

    return findings


def main():
    # 1. Wczytaj dane referencyjne
    print("Wczytywanie diversevul.json...")
    dv_lookup = load_diversevul_data(DIVERSEVUL_FILE)

    # 2. Wczytaj raport Corgea
    print("Wczytywanie raportu CSV...")
    corgea_map = load_corgea_findings(REPORT_CSV_FILE)

    results = []

    # 3. Pobierz listę plików .c w katalogu
    source_files = glob.glob(os.path.join(SOURCE_DIR, "*.c"))
    print(f"Znaleziono {len(source_files)} plików .c do analizy.")

    for file_path in source_files:
        filename = os.path.basename(file_path)

        # Parsowanie nazwy pliku: Project_GitHash_BigHash.c
        # split('_') dzieli po podkreślnikach.
        # [-1] bierze ostatni element (BigHash.c)
        try:
            name_part = os.path.splitext(filename)[0]  # Usuwa .c
            parts = name_part.split("_")

            # Identyfikatorem jest ostatnia część nazwy (BigHash)
            file_hash_id = parts[-1]

            # Pobieramy dane z DiverseVul
            dv_data = dv_lookup.get(file_hash_id, {})

            # Pobieramy dane z raportu Corgea
            # Konwertujemy set na listę dla JSON
            found_cwes = list(corgea_map.get(filename, []))

            # Jeśli w DV nie ma projektu, próbujemy go zgadnąć z nazwy pliku (wszystko przed hashami)
            # parts[-2] to git hash, parts[-1] to big hash, reszta to nazwa
            project_name = dv_data.get("project")
            if not project_name or project_name == "unknown":
                if len(parts) >= 3:
                    project_name = "_".join(parts[:-2])
                else:
                    project_name = "unknown"

            entry = {
                "id": file_hash_id,
                "project": project_name,
                "cwes": dv_data.get("cwes", []),  # Prawdziwe CWE z DiverseVul
                "cwes_corgea": found_cwes,  # Znalezione przez narzędzie
            }

            results.append(entry)

        except Exception as e:
            print(f"Błąd przy przetwarzaniu pliku {filename}: {e}")

    # 4. Zapisz wynik
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nGotowe! Przeanalizowano {len(results)} plików.")
    print(f"Wyniki zapisano w: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
