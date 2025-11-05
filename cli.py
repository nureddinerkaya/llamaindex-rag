import os
import sys
import re

MODELS = {
    "1": "bge (BAAI/bge-small-en)",
    "2": "m3 (m3)",
    "3": "multilingual-intfloat (intfloat/multilingual-embedding)",
}

EXTENSIONS_TO_SEARCH = [".py", ".txt", ".md", ".csv", ".json"]


def choose_model():
    print("Kullanılabilir modeller:")
    for k, v in MODELS.items():
        print(f"  {k}. {v}")
    choice = input("Model numarasını veya ismini girin (ör. 1 veya bge): ").strip()
    # Accept number or name
    if choice in MODELS:
        return MODELS[choice]
    # try matching by substring
    for v in MODELS.values():
        if choice.lower() in v.lower():
            return v
    print("Bilinmeyen model, varsayılan olarak 'bge' seçiliyor.")
    return MODELS["1"]


def confirm_action():
    while True:
        action = input("Dosya yükleyecek misiniz yoksa sorgu mu yapacaksınız? (dosya/sorgu): ").strip().lower()
        if action in ("dosya", "sorgu"):
            return action
        print("Lütfen 'dosya' veya 'sorgu' yazın.")


def validate_file_in_cwd(filename: str) -> str:
    # The user said files will be in same directory as py files (project root). Accept relative path.
    candidate = os.path.join(os.getcwd(), filename)
    if os.path.exists(candidate) and os.path.isfile(candidate):
        return candidate
    # also check data/ folder
    candidate2 = os.path.join(os.getcwd(), "data", filename)
    if os.path.exists(candidate2) and os.path.isfile(candidate2):
        return candidate2
    raise FileNotFoundError(f"Dosya bulunamadı: {filename}")


def handle_upload():
    filename = input("Yüklenecek dosyanın adını girin (aynı dizinde olmalı): ").strip()
    try:
        path = validate_file_in_cwd(filename)
        print(f"Başarılı: Dosya bulundu -> {path}")
        # Here you could add code to actually process/upload the file (indexing, embedding, etc.)
    except FileNotFoundError as e:
        print(str(e))


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def score_text(text: str, query_tokens):
    tokens = tokenize(text)
    if not tokens:
        return 0
    # simple bag-of-words count
    score = 0
    token_set = set(tokens)
    for qt in query_tokens:
        if qt in token_set:
            score += tokens.count(qt)
    return score


def snippet_for_match(text: str, query):
    lower = text.lower()
    idx = lower.find(query.lower())
    if idx == -1:
        # return the beginning
        return text[:200].replace('\n', ' ')
    start = max(0, idx - 60)
    end = min(len(text), idx + 140)
    return text[start:end].replace('\n', ' ')


def collect_files_to_search():
    files = []
    for root, dirs, filenames in os.walk(os.getcwd()):
        # skip __pycache__ and .venv-like folders
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", "venv", "env")]
        for fn in filenames:
            if any(fn.endswith(ext) for ext in EXTENSIONS_TO_SEARCH):
                files.append(os.path.join(root, fn))
    return files


def handle_query():
    query = input("Sorgu metnini girin: ").strip()
    if not query:
        print("Boş sorgu verildi. Çıkılıyor.")
        return
    try:
        top_k = int(input("Top-k kaç olsun? (ör. 3): ").strip())
        if top_k <= 0:
            raise ValueError()
    except ValueError:
        print("Geçersiz top-k; varsayılan 3 kullanılacak.")
        top_k = 3

    query_tokens = tokenize(query)
    files = collect_files_to_search()
    results = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        sc = score_text(text, query_tokens)
        if sc > 0:
            snip = snippet_for_match(text, query)
            results.append((sc, fpath, snip))

    if not results:
        print("Sorguyla eşleşen sonuç bulunamadı.")
        return

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"Üst {top_k} sonuç:")
    for i, (sc, fpath, snip) in enumerate(results[:top_k], start=1):
        print("\n---")
        print(f"{i}. Dosya: {os.path.relpath(fpath, os.getcwd())}")
        print(f"   Skor: {sc}")
        print(f"   Snippet: {snip}")


def main():
    print("Basit CLI - Model seçimi, dosya yükleme veya sorgu")
    model = choose_model()
    print(f"Seçilen model: {model}")
    action = confirm_action()
    if action == "dosya":
        handle_upload()
    else:
        handle_query()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından iptal edildi.")
