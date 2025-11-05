# LlamaIndex RAG CLI

Bu proje, LlamaIndex ve Qdrant kullanarak bir RAG (Retrieval-Augmented Generation) CLI uygulaması sağlar.

## Kurulum

1. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Qdrant'ı Docker ile çalıştırın:
```bash
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

## Kullanım

CLI uygulamasını çalıştırın:
```bash
python3 cli.py
```

### CLI Akışı

1. **Embedding Model Seçimi**: 
   - `bge-m3` (BAAI/bge-m3)
   - `multilingual-intfloat` (intfloat/multilingual-e5-large)

2. **İşlem Seçimi**:
   - **Dosya Yükleme (1)**: Bir PDF dosyasını vektör veritabanına yükler
   - **Sorgu (2)**: Vektör veritabanında arama yapar

3. **Dosya Yükleme**:
   - Dosya ismini girin (dosya aynı dizinde veya `data/` klasöründe olmalı)
   - Uygulama dosyayı işler ve başarı mesajı gösterir

4. **Sorgu**:
   - Sorgunuzu girin
   - Kaç sonuç istediğinizi belirtin (top-k)
   - Sonuçlar benzerlik skorları ile gösterilir

## Örnek Kullanım

```bash
$ python3 cli.py

==================================================
 RAG CLI Application - LlamaIndex + Qdrant
==================================================

=== Embedding Model Selection ===
1. bge-m3 (BAAI/bge-m3)
2. multilingual-intfloat (intfloat/multilingual-e5-large)
===================================

Select embedding model (1-2): 1

✓ Selected model: bge-m3

=== Action Selection ===
1. Upload document
2. Query documents
=========================

Select action (1-2): 1

=== Document Upload ===
Enter filename (file should be in the same directory): llama2.pdf

Uploading document: llama2.pdf
Please wait, this may take a moment...

✓ SUCCESS: Document uploaded and indexed successfully!
```

## Dosya Yapısı

- `cli.py` - Ana CLI uygulaması
- `yukleme.py` - Döküman yükleme modülü
- `qdrant_connection.py` - Qdrant bağlantı yönetimi
- `retriever.py` - Vektör veritabanı sorgu modülü
- `sorgulama.py` - Örnek sorgu betiği
- `data/` - PDF dosyaları için klasör

## Notlar

- İlk çalıştırmada embedding modelleri HuggingFace'den indirilecektir (büyük dosyalar olabilir)
- Qdrant'ın çalışır durumda olması gerekir
- PDF dosyaları `data/` klasöründe veya ana dizinde olmalıdır
