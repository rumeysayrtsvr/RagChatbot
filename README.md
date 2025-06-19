# Kadına Yönelik Şiddet ve Destek Chatbot (RAG)

Bu proje, kadına yönelik şiddet, yasal haklar ve psikolojik destek konularında bilgi sağlayan bir chatbot uygulamasıdır. Retrieval-Augmented Generation (RAG) teknolojisini kullanarak, web sayfalarından ve PDF'lerden toplanan bilgileri kullanıcı sorularına yanıt vermek için kullanır.

## Özellikler

- Kadına yönelik şiddet konusunda bilgilendirme
- Yasal haklar ve koruma tedbirleri hakkında bilgi
- Başvurulabilecek kurumlar ve iletişim bilgileri
- Psikolojik destek ve travma sonrası süreçler
- Güvenlik planlaması ve önlemler
- Sohbet geçmişini hatırlama ve bağlam koruma
- Otomatik veri güncelleme sistemi

## Teknoloji

- **RAG (Retrieval-Augmented Generation)**: Vektör veritabanından ilgili bilgileri alarak doğru ve güncel yanıtlar üretir
- **OpenAI GPT-4o**: Kullanıcı sorularına yanıt üretmek için kullanılır
- **Chroma Vektör Veritabanı**: Web sayfalarından ve PDF'lerden toplanan bilgileri saklar
- **Streamlit**: Kullanıcı arayüzü için kullanılır

## Dosya Yapısı

```
./
├── .env                      # API anahtarları ve yapılandırma
├── README.md                 # Bu dosya
├── app.py                    # Ana uygulama (Streamlit)
├── prepare_data.py           # Veri hazırlama ve vektör veritabanı oluşturma
├── rag_chatbot.py            # RAG Chatbot sınıfı
├── scheduler.py              # Otomatik veri güncelleme zamanlayıcısı
├── test_chatbot.py           # Chatbot doğruluk testi
├── requirements.txt          # Bağımlılıklar
├── data/                     # Veri dosyaları
│   └── pdfs/                 # PDF dosyaları
├── logs/                     # Log dosyaları
└── vector_store/            # Vektör veritabanı dosyaları
    └── women_rights/        # Kadın hakları koleksiyonu
```

## Kurulum

1. Gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

2. `.env` dosyasını oluşturun ve OpenAI API anahtarınızı ekleyin:

```
OPENAI_API_KEY=your_api_key_here
```

## Kullanım

1. Vektör veritabanını oluşturun:

```bash
python prepare_data.py
```

2. Chatbot uygulamasını başlatın:

```bash
streamlit run app.py
```

3. Otomatik veri güncelleme zamanlayıcısını başlatın (opsiyonel):

```bash
python scheduler.py
```

## Test

Chatbot'un doğruluk oranını test etmek için:

```bash
python test_chatbot.py
```
