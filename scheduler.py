import schedule
import time
import logging
from pathlib import Path
import os
import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Gelişmiş loglama ayarları
logging.basicConfig(
    filename=LOGS_DIR / 'scheduler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Konsola da log yazdırmak için
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

def job_web_scraper():
    """Web sitelerinden veri çekme ve vektör veritabanını güncelleme işi"""
    from prepare_data import DataLoader, urls, PDF_DIR, VECTOR_STORE_DIR
    from rag_chatbot import RAGChatbot
    
    start_time = time.time()
    job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"[{job_id}] Web scraping işi başlatılıyor")
    print(f"\n### Web scraping başlatılıyor (Job ID: {job_id}) ###")
    
    try:
        # OpenAI API anahtarını kontrol et
        if os.getenv("OPENAI_API_KEY"):
            embedding_model = "text-embedding-3-small"
            use_openai = True
            logging.info(f"OpenAI embedding modeli kullanılıyor: {embedding_model}")
        else:
            embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            use_openai = False
            logging.info(f"HuggingFace embedding modeli kullanılıyor: {embedding_model}")
        
        # Web ve PDF dokümanlarını yükle
        data_loader = DataLoader(
            pdf_dir=str(PDF_DIR),
            urls=urls,
            vector_store_path=str(VECTOR_STORE_DIR),
            embedding_model=embedding_model,
            use_openai=use_openai
        )
        
        # Web içeriğini işle
        web_docs = data_loader.load_web_content()
        logging.info(f"[{job_id}] Toplam {len(web_docs)} web dokümanı işlendi")
        
        # PDF'leri işle
        pdf_docs = data_loader.load_pdfs()
        logging.info(f"[{job_id}] Toplam {len(pdf_docs)} PDF dokümanı işlendi")
        
        # Tüm belgeleri birleştir
        all_docs = web_docs + pdf_docs
        logging.info(f"[{job_id}] Toplam {len(all_docs)} doküman işlendi")
        
        # RAG Chatbot ile vektör veritabanını güncelle
        rag_chatbot = RAGChatbot(
            vector_store_path=str(VECTOR_STORE_DIR),
            embedding_model=embedding_model,
            vector_db_type="chroma",
            collection_name="women_rights"
        )
        
        # Vektör veritabanını güncelle
        rag_chatbot.create_vector_store(all_docs)
        logging.info(f"[{job_id}] Vektör veritabanı güncellendi")
        
        elapsed_time = time.time() - start_time
        logging.info(f"[{job_id}] Web scraping tamamlandı. Süre: {elapsed_time:.2f} saniye")
        print(f"### Web scraping tamamlandı. Süre: {elapsed_time:.2f} saniye ###")
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_details = traceback.format_exc()
        logging.error(f"[{job_id}] Web scraping hatası ({elapsed_time:.2f}s): {str(e)}")
        logging.error(f"[{job_id}] Hata detayları: {error_details}")
        print(f"### HATA: {str(e)} ###")
        traceback.print_exc()
        
        return False

def start_scheduler(web_scraping_interval="03:00", run_immediately=False):
    """
    Zamanlayıcıyı başlatır ve belirtilen aralıklarla işleri çalıştırır.
    
    Args:
        web_scraping_interval: Web scraping işinin çalışma saati (varsayılan: "03:00")
        model_update_day: Model güncelleme işinin çalışacağı gün (varsayılan: "sunday")
        model_update_time: Model güncelleme işinin çalışma saati (varsayılan: "04:00")
        run_immediately: True ise, zamanlayıcı başladığında işleri hemen çalıştırır
    """
    print(f"\n=== Kadın Destek Chatbot Zamanlayıcısı Başlatılıyor (v1.0) ===\n")
    logging.info("Zamanlayıcı başlatılıyor")
    
    # Zamanlayıcı ayarları
    schedule.clear()  # Önceki zamanlayıcıları temizle
    
    # Web scraping işini planla
    schedule.every().day.at(web_scraping_interval).do(job_web_scraper)
    
    # Model güncelleme işi kaldırıldı çünkü artık fine-tuning kullanmıyoruz
    
    # Zamanlayıcı bilgilerini göster
    print("Zamanlayıcı çalışıyor:")
    print(f"- Web scraping: Her gün {web_scraping_interval}")
    print("\nSistem arka planda çalışacak ve otomatik olarak güncellenecek.")
    print("Çıkmak için Ctrl+C tuşlarına basın")
    
    # Hemen çalıştırma seçeneği
    if run_immediately:
        print("\nBaşlangıç işleri çalıştırılıyor...")
        job_web_scraper()
    
    # Ana döngü
    try:
        last_health_check = time.time()
        while True:
            schedule.run_pending()
            
            # Her 6 saatte bir sağlık kontrolü yap
            current_time = time.time()
            if current_time - last_health_check > 6 * 3600:  # 6 saat
                logging.info("Zamanlayıcı sağlıklı çalışıyor")
                last_health_check = current_time
            
            time.sleep(60)  # 1 dakika bekle
    except KeyboardInterrupt:
        print("\nZamanlayıcı durduruldu")
        logging.info("Zamanlayıcı kullanıcı tarafından durduruldu")
    except Exception as e:
        error_details = traceback.format_exc()
        logging.error(f"Zamanlayıcı hatası: {str(e)}")
        logging.error(f"Hata detayları: {error_details}")
        print(f"Zamanlayıcı hatası: {str(e)}")
        
        # Hata durumunda yeniden başlatma
        print("Zamanlayıcı yeniden başlatılıyor...")
        time.sleep(10)
        start_scheduler(web_scraping_interval, model_update_day, model_update_time, False)

if __name__ == "__main__":
    import argparse
    
    # Komut satırı argümanlarını ayarla
    parser = argparse.ArgumentParser(description="Kadın Destek Chatbot Zamanlayıcısı")
    parser.add_argument("--web-time", default="03:00", help="Web scraping saati (varsayılan: 03:00)")
    # Model güncelleme parametreleri kaldırıldı çünkü artık fine-tuning kullanmıyoruz
    parser.add_argument("--run-now", action="store_true", help="Zamanlayıcıyı başlatırken işleri hemen çalıştır")
    
    args = parser.parse_args()
    
    # Zamanlayıcıyı başlat
    start_scheduler(
        web_scraping_interval=args.web_time,
        run_immediately=args.run_now
    )
