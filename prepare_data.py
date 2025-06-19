#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Hazırlama Modülü

Bu modül, kadına yönelik şiddet, yasal haklar ve psikolojik destek konularında
veri toplamak ve işlemek için kullanılır. Web sayfaları, PDF dosyaları ve JSON
dosyalarından veri çeker ve bu verileri vektör veritabanında saklar.

Kullanım:
    python prepare_data.py --verbose
"""

import os
import sys
import json
import logging
import re
import warnings
import urllib3
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from web_scraper import WebScraper
from rag_chatbot import RAGChatbot

# .env dosyasını yükle
load_dotenv()

# Uyarıları kapat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Proje dizinleri
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"  # PDF'ler data/pdfs dizininde
JSON_DIR = DATA_DIR  # JSON dosyaları doğrudan data dizininde
LOGS_DIR = PROJECT_ROOT / "logs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
DEFAULT_COLLECTION_NAME = "women_rights"


def configure_logging(verbose: bool = False, log_file: str = "prepare_data.log"):
    """
    Logging ayarlarını yapılandırır.
    
    Args:
        verbose: Detaylı çıktı gösterilsin mi?
        log_file: Log dosyasının adı
    """
    # Log seviyesini ayarla
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Root logger'ı yapılandır
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Mevcut handler'ları temizle
    for handler in logger.handlers[:]:  
        logger.removeHandler(handler)
    
    # Konsol handler'ı ekle
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Dosya handler'ı ekle
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Requests ve urllib3 loglarını kısıtla
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Dizinleri oluştur
for dir_path in [DATA_DIR, PDF_DIR, LOGS_DIR, VECTOR_STORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# URL listesi
URL_LIST = [
    "https://morcati.org.tr/siddete-ugradiginizda-neler-yapabilirsiniz/",
    "https://morcati.org.tr/siddete-ugradiginizda-neler-yapabilirsiniz/#sikayetci-olmak-isterseniz",
    "https://siginaksizbirdunya.org/siddet-onleme-ve-izleme-merkezleri-sonim-iletisim-bilgileri/",
    "https://siginaksizbirdunya.org/siddete-ugradiginizda-basvurabileceginiz-kadin-orgutleri/",
    "https://www.anayasa.gov.tr/tr/mevzuat/anayasa/",
    "https://www.asayis.pol.tr/cinsel-suclar",
    "https://datem.com.tr/blog/kadina-yonelik-siddet-ve-psikolojik-etkileri/",
    "https://kadem.org.tr/siddet-nedir-ve-siddete-karsi-kadinin-korunmasi-ve-basvuru-yollari/",
    "https://www.kadindayanismavakfi.org.tr/siddete-maruz-kaldigimizda-ne-yapmaliyiz/",
    "https://morcati.org.tr/yayinlarimiz/brosurler/kadina-yoneli-siddetle-mucadelede-siginaklar/",
    "https://morcati.org.tr/siddet-bicimleri/#israrli-takip",
    "https://evicisiddet.adalet.gov.tr/Ne_Yapabilirim.html",
    "https://psikiyatri.org.tr/halka-yonelik/28/travma-sonrasi-stres-bozuklugu",
    "https://www.adnancoban.com.tr/istismar-ve-travma",
    "https://www.umapsikoloji.com/post/psikolojik-ilk-yardim",
    "https://www.nirengidernegi.org/tr/siddete-maruz-kaldim-ne-yapmaliyim/",
    "https://barandogan.av.tr/blog/ceza-hukuku/cinsel-suclarda-magdur-sikayetci-beyaninin-delil-degeri.html",
    "https://cinselsiddetlemucadele.org/2020/05/27/cinsel-istismar-hakkinda-yanlis-bilinenler/",
    "https://aile.gov.tr/iletisim/bakanlik-iletisim-bilgileri/sonim/",
    "https://kararaldim.org/siddete-maruz-kaldiginizda-ulasabileceginiz-kurumlar/",
    "https://www.aile.gov.tr/sss/kadinin-statusu-genel-mudurlugu/",
    "https://kadinininsanhaklari.org/destek-almak-icin-basvurulabilecek-kurumlar/"
]

# Logging ayarları configure_logging fonksiyonu ile yapılıyor


def detect_content_type(url: str) -> Dict[str, Any]:
    """
    URL'nin içerik türünü tespit eder ve uygun yapılandırma döndürür.
    
    Args:
        url: İçerik türü tespit edilecek URL
        
    Returns:
        Yapılandırma bilgilerini içeren sözlük
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    domain = parsed_url.netloc.lower()
    
    # SSS sayfaları için ön kontrol
    if "/sss/" in path or "/faq/" in path or "sikca-sorulan-sorular" in path:
        return {
            "data_type": "faq",
            "question_selector": ".soru-baslik, .faq-question, .accordion-header, dt, h3, h4",
            "answer_selector": ".soru-cevap, .faq-answer, .accordion-body, dd, p"
        }
    
    # Belirli domainler için özel yapılandırmalar
    if "morcati.org.tr" in domain:
        return {
            "data_type": "content",
            "selectors": {
                "title": "h1.page-title, .entry-title",
                "content": ".entry-content, .page-content"
            }
        }
    elif "siginaksizbirdunya.org" in domain:
        return {
            "data_type": "content",
            "selectors": {
                "title": "h1.entry-title",
                "content": ".entry-content"
            }
        }
    elif "aile.gov.tr" in domain:
        if "/sss/" in path:
            return {
                "data_type": "faq",
                "question_selector": ".panel-heading h4",
                "answer_selector": ".panel-body"
            }
        else:
            return {
                "data_type": "content",
                "selectors": {
                    "title": ".page-title h1",
                    "content": ".content-body"
                }
            }
    
    # Genel içerik türü tespiti için web sayfasını analiz et
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        
        if response.status_code != 200:
            return {"data_type": "content", "selectors": {"title": "h1", "content": ".content, article, .entry-content"}}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tablo kontrolü
        tables = soup.find_all('table')
        if len(tables) > 0:
            return {"data_type": "table", "table_selector": "table"}
        
        # Accordion kontrolü
        accordions = soup.select('.accordion, .accordion-item, .panel-group, .faq-item')
        if accordions:
            return {
                "data_type": "content",
                "force_selenium": True,
                "click_selectors": [".accordion-header, .accordion-button, .panel-heading"],
                "wait_selectors": [".accordion-body, .panel-body"],
                "selectors": {"title": "h1", "content": ".content, article, .entry-content"}
            }
        
        # Soru-cevap kontrolü
        headings = soup.find_all(['h2', 'h3', 'h4', 'dt'])
        if sum(1 for h in headings if any(p in h.get_text().lower() for p in ['soru', '?'])) >= 2:
            return {
                "data_type": "faq",
                "question_selector": "h2, h3, h4, dt",
                "answer_selector": "p, div, dd"
            }
        
        # Liste kontrolü
        lists = soup.find_all(['ul', 'ol'])
        if len(lists) >= 2:
            return {"data_type": "list", "list_selector": "ul li, ol li, .list-item"}
        
        # Varsayılan içerik
        return {
            "data_type": "content",
            "selectors": {
                "title": "h1, .page-title, .entry-title, .article-title",
                "content": ".content, .entry-content, .page-content, article, main"
            }
        }
        
    except Exception as e:
        logging.error(f"URL analizi hatası: {url} - {str(e)}")
        return {"data_type": "content", "selectors": {"title": "h1", "content": ".content, article, .entry-content"}}


def scrape_web_content(urls: List[str], verbose: bool = False) -> List[Document]:
    """
    Web sayfalarından veri çeker ve Document nesneleri listesi olarak döndürür.
    
    Args:
        urls: Veri çekilecek URL'lerin listesi
        verbose: Detaylı çıktı gösterilsin mi?
        
    Returns:
        Document nesneleri listesi
    """
    logger = logging.getLogger(__name__)
    web_scraper = WebScraper(urls, log_level=logging.INFO if verbose else logging.WARNING)
    documents = []
    
    if not urls:
        logger.warning("URL listesi boş")
        return documents
    
    logger.info(f"Toplam {len(urls)} URL işlenecek")
    
    # Her URL için ayrı ayrı işlem yap
    for url in urls:
        try:
            logger.info(f"Veri çekiliyor: {url}")
            parsed_url = urlparse(url)
            
            # İçerik türünü tespit et
            config = detect_content_type(url)
            use_selenium = config.get("force_selenium", False)
            data_type = config.get("data_type", "content")
            
            logger.info(f"URL içerik türü: {data_type}")
            
            # İçeriği çekmeyi dene - önce yapılandırılmış içerik çekmeyi dene
            data = None
            try:
                data = web_scraper.scrape_structured_content(url, config, use_selenium=use_selenium)
                logger.info(f"Yapılandırılmış veri çekildi: {url}")
            except Exception as e:
                logger.warning(f"Yapılandırılmış veri çekilemedi: {url} - {str(e)}")
                data = None
            
            # Yapılandırılmış içerik çekilemediyse basit yöntemi dene
            if not data:
                logger.warning(f"Alternatif yöntem deneniyor: {url}")
                try:
                    content = web_scraper.scrape_url(url)
                    if content and content.strip():
                        # Temel metadata oluştur
                        metadata = {
                            "source": url, 
                            "title": url.split("/")[-1] or "Web Sayfası", 
                            "date": datetime.now().strftime("%Y-%m-%d"), 
                            "type": "web",
                            "data_type": "basic_content",
                            "domain": parsed_url.netloc
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                        logger.info(f"Basit yöntemle veri çekildi: {url} - {len(content)} karakter")
                    else:
                        logger.error(f"URL'den veri çekilemedi: {url}")
                except Exception as e:
                    logger.error(f"Alternatif yöntem başarısız: {url} - {str(e)}")
                continue
            
            # Yapılandırılmış içeriği işle
            content = ""
            metadata = {
                "source": url, 
                "date": datetime.now().strftime("%Y-%m-%d"), 
                "type": "web",
                "data_type": data_type,
                "domain": parsed_url.netloc
            }
            
            # İçerik türüne göre işleme
            if isinstance(data, str):
                # Düz metin içerik
                content = data
                metadata["title"] = url.split("/")[-1] or "Web Sayfası"
            elif isinstance(data, dict):
                # Başlık bilgisini ekle
                if "title" in data and data["title"]:
                    metadata["title"] = data["title"]
                else:
                    metadata["title"] = url.split("/")[-1] or "Web Sayfası"
                
                # Veri tipine göre işleme
                if data_type == "faq" and "faq_items" in data and data["faq_items"]:
                    try:
                        content = "\n\n".join(f"Soru: {item['question']}\n\nCevap: {item['answer']}" for item in data["faq_items"])
                        metadata["faq_count"] = len(data["faq_items"])
                    except Exception as e:
                        logger.error(f"FAQ verisi işlenirken hata: {url} - {str(e)}")
                        content = str(data.get("faq_items", ""))[:1000]
                        
                elif data_type == "table" and "tables" in data and data["tables"]:
                    try:
                        table_contents = []
                        for i, table in enumerate(data["tables"]):
                            table_content = f"Tablo {i+1}:\n"
                            for row in table:
                                table_content += " | ".join(str(cell) for cell in row) + "\n"
                            table_contents.append(table_content)
                        content = "\n\n".join(table_contents)
                        metadata["table_count"] = len(data["tables"])
                    except Exception as e:
                        logger.error(f"Tablo verisi işlenirken hata: {url} - {str(e)}")
                        content = str(data.get("tables", ""))[:1000]
                        
                elif data_type == "list" and "list_items" in data and data["list_items"]:
                    try:
                        content = "\n".join(f"- {item}" for item in data["list_items"])
                        metadata["list_count"] = len(data["list_items"])
                    except Exception as e:
                        logger.error(f"Liste verisi işlenirken hata: {url} - {str(e)}")
                        content = str(data.get("list_items", ""))[:1000]
                        
                elif "title" in data or "content" in data:
                    parts = []
                    if "title" in data and data["title"]:
                        parts.append(f"Başlık: {data['title']}")
                    if "content" in data and data["content"]:
                        parts.append(f"\n\n{data['content']}")
                    content = "\n".join(parts)
                else:
                    # Genel sözlük içeriği
                    content = "\n".join(f"{key}: {value}" for key, value in data.items() if key not in ["url", "data_type", "error"])
            
            # İçerik kontrolü ve doküman oluşturma
            if content and content.strip():
                # İçeriği temizle
                content = content.strip()
                # Çok uzun boşlukları temizle
                content = re.sub(r'\n{3,}', '\n\n', content)
                # Doküman oluştur
                documents.append(Document(page_content=content, metadata=metadata))
                logger.info(f"URL içeriği başarıyla işlendi: {url} - {len(content)} karakter")
            else:
                logger.warning(f"URL içeriği boş: {url}")
                
        except Exception as e:
            logger.error(f"URL işlenirken hata: {url} - {str(e)}")
    
    logger.info(f"Toplam {len(documents)} web dokümanı işlendi")
    return documents


def process_pdf_files(pdf_dir: str, verbose: bool = False) -> List[Document]:
    """
    PDF dosyalarını yükler ve parçalara böler.
    
    Args:
        pdf_dir: PDF dosyalarının bulunduğu dizin
        verbose: Detaylı çıktı gösterilsin mi?
        
    Returns:
        Document nesneleri listesi
    """
    logger = logging.getLogger(__name__)
    pdf_dir_path = Path(pdf_dir)
    
    # Dizin var mı kontrol et
    if not pdf_dir_path.exists() or not pdf_dir_path.is_dir():
        logger.error(f"PDF dizini bulunamadı veya geçerli bir dizin değil: {pdf_dir}")
        return []
    
    # PDF dosyalarını bul
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"PDF dizininde dosya bulunamadı: {pdf_dir}")
        return []

    logger.info(f"Toplam {len(pdf_files)} PDF dosyası bulundu")
    documents = []
    
    # Her PDF dosyasını işle
    for pdf_file in pdf_files:
        # Dosya boyutunu kontrol et
        if pdf_file.stat().st_size == 0:
            logger.warning(f"PDF dosyası boş: {pdf_file.name}")
            continue
            
        try:
            logger.info(f"PDF dosyası yükleniyor: {pdf_file.name}")
            
            # PDF dosyasını yükle
            loader = PyPDFLoader(str(pdf_file))
            pdf_pages = loader.load()
            
            if not pdf_pages:
                logger.warning(f"PDF dosyasından sayfa yüklenemedi: {pdf_file.name}")
                continue
                
            # Her sayfayı işle ve metadata ekle
            for i, page in enumerate(pdf_pages):
                try:
                    # Metadata'yı zenginleştir
                    page.metadata["source"] = str(pdf_file)
                    page.metadata["file_name"] = pdf_file.name
                    page.metadata["type"] = "pdf"
                    page.metadata["date"] = datetime.now().strftime("%Y-%m-%d")
                    page.metadata["title"] = pdf_file.stem
                    page.metadata["page"] = i + 1
                    page.metadata["total_pages"] = len(pdf_pages)
                    
                    # İçerik kontrolü
                    if not page.page_content or not page.page_content.strip():
                        logger.warning(f"PDF sayfası boş içerik: {pdf_file.name}, sayfa {i+1}")
                        continue
                        
                    documents.append(page)
                except Exception as e:
                    logger.error(f"PDF sayfası işlenirken hata: {pdf_file.name}, sayfa {i+1} - {str(e)}")
            
            logger.info(f"PDF dosyası yüklendi: {pdf_file.name} - {len(pdf_pages)} sayfa")
            
        except Exception as e:
            logger.error(f"PDF yükleme hatası: {pdf_file.name} - {str(e)}")
    
    # Dokümanları parçalara böl
    if documents:
        try:
            # Metin bölücü oluştur
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Her parçanın maksimum karakter sayısı
                chunk_overlap=200,  # Parçalar arasındaki örtüşme miktarı
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Bölme önceliği
                length_function=len  # Uzunluk hesaplama fonksiyonu
            )
            
            # Dokümanları parçalara böl
            split_documents = text_splitter.split_documents(documents)
            logger.info(f"Toplam {len(split_documents)} PDF parçası oluşturuldu (orijinal: {len(documents)} sayfa)")
            return split_documents
        except Exception as e:
            logger.error(f"PDF parçalama hatası: {str(e)}")
            return documents
    else:
        logger.warning("PDF dokümanı bulunamadı veya yüklenemedi")
        return []


def process_json_files(json_dir: str, verbose: bool = False) -> List[Document]:
    """
    JSON dosyalarını yükler ve Document nesnelerine dönüştürür.
    
    Args:
        json_dir: JSON dosyalarının bulunduğu dizin
        verbose: Detaylı çıktı gösterilsin mi?
        
    Returns:
        Document nesneleri listesi
    """
    logger = logging.getLogger(__name__)
    json_dir_path = Path(json_dir)
    
    # Dizin var mı kontrol et
    if not json_dir_path.exists() or not json_dir_path.is_dir():
        logger.error(f"JSON dizini bulunamadı veya geçerli bir dizin değil: {json_dir}")
        return []
    
    # JSON dosyalarını bul
    json_files = list(json_dir_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"JSON dizininde dosya bulunamadı: {json_dir}")
        return []

    logger.info(f"Toplam {len(json_files)} JSON dosyası bulundu")
    documents = []
    
    # Her JSON dosyasını işle
    for json_file in json_files:
        # Dosya boyutunu kontrol et
        if json_file.stat().st_size == 0:
            logger.warning(f"JSON dosyası boş: {json_file.name}")
            continue
            
        try:
            logger.info(f"JSON dosyası yükleniyor: {json_file.name}")
            
            # JSON dosyasını oku
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON dosyası geçersiz format: {json_file.name} - {str(e)}")
                    continue
            
            # JSON içeriğini işle
            if not data:
                logger.warning(f"JSON dosyası boş içerik: {json_file.name}")
                continue
                
            # JSON türünü kontrol et ve uygun şekilde işle
            if isinstance(data, list):
                # Liste tipindeki JSON dosyaları için
                for i, item in enumerate(data):
                    try:
                        if not item:  # Boş öğe kontrolü
                            continue
                            
                        # Öğe içeriğini oluştur
                        if isinstance(item, dict):
                            content = "\n".join(f"{k}: {v}" for k, v in item.items())
                        else:
                            content = str(item)
                            
                        # Metadata oluştur
                        metadata = {
                            "source": str(json_file),
                            "file_name": json_file.name,
                            "type": "json",
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "title": f"{json_file.stem} - Öğe {i+1}",
                            "item_index": i,
                            "total_items": len(data)
                        }
                        
                        # Doküman oluştur
                        documents.append(Document(page_content=content, metadata=metadata))
                    except Exception as e:
                        logger.error(f"JSON liste öğesi işlenirken hata: {json_file.name}, öğe {i} - {str(e)}")
                        
                logger.info(f"JSON liste dosyası işlendi: {json_file.name} - {len(data)} öğe")
                
            elif isinstance(data, dict):
                # Sözlük tipindeki JSON dosyaları için
                try:
                    # Özel JSON dosyaları için işleme
                    if json_file.name == "siginma_evleri.json":
                        # Sığınma evleri JSON dosyası için özel işleme
                        for il, bilgiler in data.items():
                            if isinstance(bilgiler, dict):
                                content = f"\u0130l: {il}\n"
                                for kurum_turu, kurum_bilgileri in bilgiler.items():
                                    content += f"\n{kurum_turu}:\n"
                                    if isinstance(kurum_bilgileri, dict):
                                        for bilgi_turu, deger in kurum_bilgileri.items():
                                            content += f"  {bilgi_turu}: {deger}\n"
                                    elif isinstance(kurum_bilgileri, list):
                                        for bilgi in kurum_bilgileri:
                                            content += f"  - {bilgi}\n"
                                    else:
                                        content += f"  {kurum_bilgileri}\n"
                                
                                metadata = {
                                    "source": str(json_file),
                                    "file_name": json_file.name,
                                    "type": "json",
                                    "subtype": "siginma_evi",
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                    "title": f"Sığınma Evleri - {il}",
                                    "il": il
                                }
                                
                                documents.append(Document(page_content=content, metadata=metadata))
                        
                        logger.info(f"Sığınma evleri JSON dosyası işlendi: {json_file.name} - {len(data)} il")
                        
                    elif json_file.name == "kadin_danisma_merkezleri.json":
                        # Kadın danışma merkezleri JSON dosyası için özel işleme
                        for merkez_turu, merkezler in data.items():
                            if isinstance(merkezler, list):
                                for merkez in merkezler:
                                    if isinstance(merkez, dict):
                                        content = f"Merkez Türü: {merkez_turu}\n"
                                        for bilgi_turu, deger in merkez.items():
                                            if isinstance(deger, list):
                                                content += f"\n{bilgi_turu}:\n"
                                                for item in deger:
                                                    content += f"  - {item}\n"
                                            else:
                                                content += f"\n{bilgi_turu}: {deger}\n"
                                        
                                        metadata = {
                                            "source": str(json_file),
                                            "file_name": json_file.name,
                                            "type": "json",
                                            "subtype": "danisma_merkezi",
                                            "date": datetime.now().strftime("%Y-%m-%d"),
                                            "title": merkez.get("adi", f"Kadın Danışma Merkezi"),
                                            "merkez_turu": merkez_turu,
                                            "sehir": merkez.get("sehir", "")
                                        }
                                        
                                        documents.append(Document(page_content=content, metadata=metadata))
                        
                        logger.info(f"Kadın danışma merkezleri JSON dosyası işlendi: {json_file.name}")
                        
                    elif json_file.name == "pdf_metadata.json":
                        # PDF metadata JSON dosyası için özel işleme
                        for pdf_name, pdf_info in data.items():
                            if isinstance(pdf_info, dict):
                                content = f"PDF Dosyası: {pdf_name}\n"
                                for bilgi_turu, deger in pdf_info.items():
                                    content += f"\n{bilgi_turu}: {deger}\n"
                                
                                metadata = {
                                    "source": str(json_file),
                                    "file_name": json_file.name,
                                    "type": "json",
                                    "subtype": "pdf_metadata",
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                    "title": f"PDF Metadata - {pdf_name}",
                                    "pdf_name": pdf_name
                                }
                                
                                documents.append(Document(page_content=content, metadata=metadata))
                        
                        logger.info(f"PDF metadata JSON dosyası işlendi: {json_file.name} - {len(data)} PDF dosyası")
                    
                    else:
                        # Genel sözlük işleme
                        content = "\n".join(f"{k}: {v}" for k, v in data.items())
                        metadata = {
                            "source": str(json_file),
                            "file_name": json_file.name,
                            "type": "json",
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "title": json_file.stem
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                        logger.info(f"JSON sözlük dosyası işlendi: {json_file.name}")
                        
                except Exception as e:
                    logger.error(f"JSON sözlük işlenirken hata: {json_file.name} - {str(e)}")
            else:
                # Diğer JSON türleri için
                content = str(data)
                metadata = {
                    "source": str(json_file),
                    "file_name": json_file.name,
                    "type": "json",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "title": json_file.stem
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                logger.info(f"JSON dosyası işlendi: {json_file.name}")
                
        except Exception as e:
            logger.error(f"JSON dosyası işlenirken hata: {json_file.name} - {str(e)}")
    
    # Dokümanları parçalara böl
    if documents:
        try:
            # Metin bölücü oluştur
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            # Dokümanları parçalara böl
            split_documents = text_splitter.split_documents(documents)
            logger.info(f"Toplam {len(split_documents)} JSON parçası oluşturuldu (orijinal: {len(documents)} doküman)")
            return split_documents
        except Exception as e:
            logger.error(f"JSON parçalama hatası: {str(e)}")
            return documents
    else:
        logger.warning("JSON dokümanı bulunamadı veya yüklenemedi")
        return []


def prepare_vector_database(documents: List[Document], vector_store_path: str, embedding_model: str = "text-embedding-3-small") -> bool:
    """
    Dokümanları kullanarak vektör veritabanı oluşturur.
    
    Args:
        documents: Vektör veritabanına eklenecek dokümanlar
        vector_store_path: Vektör veritabanının kaydedileceği dizin
        embedding_model: Kullanılacak embedding modeli
        
    Returns:
        İşlemin başarılı olup olmadığı
    """
    logger = logging.getLogger(__name__)
    
    if not documents:
        logger.error("Vektör veritabanı oluşturmak için doküman bulunamadı")
        return False
    
    logger.info(f"Vektör veritabanı oluşturuluyor: {len(documents)} doküman")
    
    try:
        # Geçersiz dokümanları filtrele
        valid_documents = []
        for doc in documents:
            if not isinstance(doc, Document):
                logger.warning(f"Geçersiz doküman tipi: {type(doc)}")
                continue
                
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(f"Boş içerikli doküman: {doc.metadata.get('source', 'Bilinmeyen')}")
                continue
                
            valid_documents.append(doc)
        
        # Geçerli doküman sayısını kontrol et
        if not valid_documents:
            logger.error("Geçerli doküman bulunamadı")
            return False
            
        logger.info(f"Toplam {len(valid_documents)} geçerli doküman işlenecek (filtre sonrası)")
        
        # Embedding modeli oluştur
        embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Vektör veritabanı dizinini oluştur
        vector_store_path = Path(vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Vektör veritabanını oluştur
        vectorstore = FAISS.from_documents(valid_documents, embeddings)
        
        # Vektör veritabanını kaydet
        vectorstore.save_local(str(vector_store_path))
        
        logger.info(f"Vektör veritabanı başarıyla oluşturuldu: {vector_store_path}")
        return True
        
    except Exception as e:
        logger.error(f"Vektör veritabanı oluşturma hatası: {str(e)}")
        return False


def main(urls_file: str = "urls.txt", 
         pdf_dir: str = str(PDF_DIR), 
         json_dir: str = str(JSON_DIR), 
         vector_store_path: str = str(VECTOR_STORE_DIR),
         embedding_model: str = "text-embedding-3-small",
         verbose: bool = False) -> bool:
    """
    Ana fonksiyon - tüm veri kaynak türlerini işler ve vektör veritabanı oluşturur.
    
    Args:
        urls_file: URL'lerin bulunduğu dosya
        pdf_dir: PDF dosyalarının bulunduğu dizin
        json_dir: JSON dosyalarının bulunduğu dizin
        vector_store_path: Vektör veritabanının kaydedileceği dizin
        embedding_model: Kullanılacak embedding modeli
        verbose: Detaylı çıktı gösterilsin mi?
        
    Returns:
        İşlemin başarılı olup olmadığı
    """
    # Logging ayarlarını yapılandır
    configure_logging(verbose=verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Veri hazırlama işlemi başlıyor")
    logger.info(f"URL dosyası: {urls_file}")
    logger.info(f"PDF dizini: {pdf_dir}")
    logger.info(f"JSON dizini: {json_dir}")
    logger.info(f"Vektör veritabanı dizini: {vector_store_path}")
    
    all_documents = []
    
    # Web içeriklerini işle - dosyada tanımlı URL listesini kullan
    try:
        # Dosyada tanımlı URL listesini kullan
        if URL_LIST:
            logger.info(f"Toplam {len(URL_LIST)} URL işlenecek")
            web_documents = scrape_web_content(URL_LIST, verbose=verbose)
            logger.info(f"Web içeriği işlendi: {len(web_documents)} doküman")
            all_documents.extend(web_documents)
        else:
            logger.warning("URL listesi boş")
    except Exception as e:
        logger.error(f"Web içeriği işleme hatası: {str(e)}")
        logger.info("Web içeriği işleme hatası nedeniyle sadece PDF ve JSON dosyaları kullanılacak")
    
    # PDF dosyalarını işle
    try:
        pdf_documents = process_pdf_files(pdf_dir, verbose=verbose)
        logger.info(f"PDF dosyaları işlendi: {len(pdf_documents)} doküman")
        all_documents.extend(pdf_documents)
    except Exception as e:
        logger.error(f"PDF işleme hatası: {str(e)}")
    
    # JSON dosyalarını işle
    try:
        json_documents = process_json_files(json_dir, verbose=verbose)
        logger.info(f"JSON dosyaları işlendi: {len(json_documents)} doküman")
        all_documents.extend(json_documents)
    except Exception as e:
        logger.error(f"JSON işleme hatası: {str(e)}")
    
    # Toplam doküman sayısını kontrol et
    if not all_documents:
        logger.error("Hiçbir doküman bulunamadı veya işlenemedi")
        return False
    
    logger.info(f"Toplam {len(all_documents)} doküman işlendi")
    
    # Vektör veritabanını oluştur
    try:
        # RAGChatbot sınıfını kullanarak vektör veritabanı oluştur
        from rag_chatbot import RAGChatbot
        
        # RAGChatbot nesnesini oluştur
        chatbot = RAGChatbot(
            vector_store_path=vector_store_path, 
            embedding_model=embedding_model,
            collection_name="women_rights"
        )
        
        # Vektör veritabanını oluştur
        if len(all_documents) > 0:
            # create_vector_store metodu ile vektör veritabanı oluştur
            chatbot.create_vector_store(all_documents)
            logger.info(f"Vektör veritabanı başarıyla oluşturuldu: {vector_store_path}")
            return True
        else:
            logger.error("Vektör veritabanı oluşturmak için doküman bulunamadı")
            return False
            
    except Exception as e:
        logger.error(f"RAGChatbot ile vektör veritabanı oluşturma hatası: {str(e)}")
        
        # Alternatif yöntem - doğrudan FAISS kullanarak oluştur
        logger.info("Alternatif yöntem deneniyor: FAISS ile doğrudan vektör veritabanı oluşturma")
        return prepare_vector_database(all_documents, vector_store_path, embedding_model)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Veri hazırlama ve vektör veritabanı oluşturma aracı")
    parser.add_argument("--urls", type=str, default="urls.txt", help="URL'lerin bulunduğu dosya")
    parser.add_argument("--pdf-dir", type=str, default="data/pdfs", help="PDF dosyalarının bulunduğu dizin")
    parser.add_argument("--json-dir", type=str, default="data", help="JSON dosyalarının bulunduğu dizin")
    parser.add_argument("--vector-store", type=str, default="./vector_store", help="Vektör veritabanının kaydedileceği dizin")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="Kullanılacak embedding modeli")
    parser.add_argument("--verbose", action="store_true", help="Detaylı çıktı göster")
    
    args = parser.parse_args()
    
    success = main(
        urls_file=args.urls,
        pdf_dir=args.pdf_dir,
        json_dir=args.json_dir,
        vector_store_path=args.vector_store,
        embedding_model=args.embedding_model,
        verbose=args.verbose
    )
    
    if success:
        print("\nVeri hazırlama ve vektör veritabanı oluşturma işlemi başarıyla tamamlandı.")
        sys.exit(0)
    else:
        print("\nVeri hazırlama ve vektör veritabanı oluşturma işlemi sırasında hatalar oluştu. Detaylar için log dosyasını kontrol edin.")
        sys.exit(1)
