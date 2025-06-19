#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
import requests
import urllib3
import warnings
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from langchain.schema import Document
from dotenv import load_dotenv

# Uyarıları kapat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class WebScraper:
    """Web sitelerinden veri çekmek için kullanılan sınıf."""
    
    def __init__(self, urls: List[str] = None, log_level: int = logging.WARNING):
        """
        WebScraper sınıfını başlat.
        
        Args:
            urls: Veri çekilecek URL'lerin listesi
            log_level: Logging seviyesi
        """
        self.urls = urls or []
        
        # Logger ayarları
        self.logger = logging.getLogger("WebScraper")
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def scrape_url(self, url: str) -> Optional[str]:
        """
        Belirtilen URL'den içeriği çeker.
        
        Args:
            url: İçeriği çekilecek URL
            
        Returns:
            str: Çekilen içerik veya None (hata durumunda)
        """
        self.logger.info(f"URL çekiliyor: {url}")
        
        # Farklı User-Agent'lar deneyelim
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Standart headers
        headers = {
            'User-Agent': user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Önce standart headers ile deneyelim
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=30)
            
            # Eğer başarılı olduysa içeriği işle
            if response.status_code == 200:
                return self._process_html_content(response.content, url)
            
            # 403 veya 406 hatası alırsak farklı User-Agent'lar deneyelim
            elif response.status_code in [403, 406]:
                self.logger.warning(f"İlk deneme başarısız ({url}): HTTP {response.status_code}, farklı User-Agent'lar deneniyor...")
                
                for user_agent in user_agents[1:]:
                    headers['User-Agent'] = user_agent
                    try:
                        response = requests.get(url, headers=headers, verify=False, timeout=30)
                        if response.status_code == 200:
                            self.logger.info(f"Farklı User-Agent ile başarılı: {user_agent}")
                            return self._process_html_content(response.content, url)
                    except Exception as e:
                        self.logger.warning(f"Alternatif User-Agent denemesi başarısız: {str(e)}")
                        continue
                
                # Hala başarısız ise, selenium ile deneme yap
                self.logger.warning(f"Tüm User-Agent'lar başarısız. Selenium ile deneniyor...")
                return self._try_selenium_fallback(url)
            
            else:
                self.logger.warning(f"URL çekilemedi ({url}): HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"URL çekilemedi ({url}): {e}")
            return None
    
    def _process_html_content(self, content, url) -> str:
        """HTML içeriğini işler ve metin olarak döndürür"""
        try:
            soup = BeautifulSoup(content, "html.parser")
            
            # Gereksiz HTML elementlerini temizle
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Metin içeriğini al
            text = soup.get_text(separator="\n")
            
            # Fazla boşlukları temizle
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            self.logger.info(f"URL başarıyla çekildi: {url}")
            return text
        except Exception as e:
            self.logger.warning(f"HTML içeriği işlenirken hata: {str(e)}")
            return None
    
    def _try_selenium_fallback(self, url) -> Optional[str]:
        """Selenium ile web sayfasını çekmeyi dener (requests başarısız olduğunda)"""
        try:
            self.logger.info(f"Selenium ile deneniyor: {url}")
            # Selenium kurulu değilse uyarı ver
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
            except ImportError:
                self.logger.warning("Selenium kurulu değil. 'pip install selenium webdriver-manager' komutu ile kurabilirsiniz.")
                return None
            
            # Chrome seçeneklerini ayarla
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # WebDriver'ı başlat
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            
            # URL'yi ziyaret et
            driver.get(url)
            
            # Sayfanın yüklenmesini bekle
            import time
            time.sleep(5)
            
            # Sayfa içeriğini al
            page_source = driver.page_source
            
            # WebDriver'ı kapat
            driver.quit()
            
            # İçeriği işle
            return self._process_html_content(page_source, url)
            
        except Exception as e:
            self.logger.warning(f"Selenium ile deneme başarısız: {str(e)}")
            return None
    
    def scrape_paginated_site_with_requests(self, site_config: Dict[str, Any]) -> List[str]:
        """
        Requests kütüphanesi kullanarak çok sayfalı bir web sitesinden veri çeker.
        
        Args:
            site_config: Site yapılandırması (base_url, param, start, end)
            
        Returns:
            list: Tüm sayfalardan çekilen içerikler
        """
        all_content = []
        
        for page in range(site_config["start"], site_config.get("end", site_config["start"] + 10) + 1):
            url = f"{site_config['base_url']}?{site_config['param']}={page}"
            self.logger.info(f"Sayfa ziyaret ediliyor: {url}")
            
            try:
                response = requests.get(url, verify=False, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    content = soup.get_text()
                    all_content.append(content)
                    self.logger.info(f"Sayfa {page} içeriği alındı. Boyut: {len(content)} karakter")
                else:
                    self.logger.warning(f"Sayfa alınamadı. Durum kodu: {response.status_code}")
                    # Eğer 404 hatası alırsak, muhtemelen son sayfayı geçmişizdir
                    if response.status_code == 404:
                        break
            except Exception as e:
                self.logger.error(f"Sayfa {page} alınırken hata: {str(e)}")
        
        return all_content
    
    def scrape_with_selenium(self, site_config: Dict[str, Any]) -> List[str]:
        """
        Selenium kullanarak çok sayfalı bir web sitesinden veri çeker.
        
        Args:
            site_config: Site yapılandırması (base_url, param, start, end, selector)
            
        Returns:
            list: Tüm sayfalardan çekilen içerikler
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import time
        except ImportError:
            self.logger.error("Selenium kütüphanesi bulunamadı. 'pip install selenium' komutu ile yükleyebilirsiniz.")
            return []
        
        all_content = []
        
        # Tarayıcı ayarları
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Tarayıcıyı görünmez modda çalıştır
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            # Sayfa sayısını otomatik tespit etme
            if "end" not in site_config and "selector" in site_config:
                driver.get(site_config["base_url"])
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, site_config["selector"]))
                )
                pagination_elements = driver.find_elements(By.CSS_SELECTOR, site_config["selector"])
                try:
                    # Son sayfa numarasını bulmaya çalış
                    page_numbers = [int(el.text) for el in pagination_elements if el.text.isdigit()]
                    if page_numbers:
                        site_config["end"] = max(page_numbers)
                    else:
                        site_config["end"] = site_config["start"] + 5  # Varsayılan olarak 5 sayfa
                except Exception as e:
                    self.logger.error(f"Sayfa sayısı tespitinde hata: {str(e)}")
                    site_config["end"] = site_config["start"] + 5
            
            # Her sayfayı ziyaret et
            for page in range(site_config["start"], site_config.get("end", site_config["start"] + 10) + 1):
                url = f"{site_config['base_url']}?{site_config['param']}={page}"
                self.logger.info(f"Sayfa ziyaret ediliyor: {url}")
                
                driver.get(url)
                time.sleep(2)  # Sayfanın yüklenmesi için bekle
                
                # Sayfadaki içeriği al
                content = driver.find_element(By.TAG_NAME, "body").text
                all_content.append(content)
                
                self.logger.info(f"Sayfa {page} içeriği alındı. Boyut: {len(content)} karakter")
            
            return all_content
        
        except Exception as e:
            self.logger.error(f"Selenium ile veri çekmede hata: {str(e)}")
            return []
        
        finally:
            if 'driver' in locals():
                driver.quit()
    
    def scrape_with_link_following(self, start_url: str, max_depth: int = 2, 
                                  max_pages: int = 50, domain_filter: str = None, 
                                  url_filter: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Başlangıç URL'inden başlayarak, sayfadaki linkleri takip ederek veri çeker.
        
        Args:
            start_url: Başlangıç URL'i
            max_depth: Maksimum takip derinliği (1: sadece başlangıç sayfası, 2: başlangıç sayfası ve onun linkleri, vb.)
            max_pages: Maksimum ziyaret edilecek sayfa sayısı
            domain_filter: Sadece bu domain içindeki linkleri takip et (None: tüm domainler)
            url_filter: Sadece bu deseni içeren URL'leri takip et (None: tüm URL'ler)
            
        Returns:
            dict: URL'lere göre içerikler
        """
        from urllib.parse import urlparse, urljoin
        import re
        
        # Ziyaret edilen URL'leri takip et
        visited_urls = set()
        # URL'leri derinliklerine göre sakla
        url_queue = [(start_url, 0)]  # (url, depth)
        # Sonuçları sakla
        results = {}
        
        # Domain filtresini ayarla
        if domain_filter is None and start_url:
            parsed_url = urlparse(start_url)
            domain_filter = parsed_url.netloc
            self.logger.info(f"Domain filtresi otomatik olarak ayarlandı: {domain_filter}")
        
        # URL filtresini regex'e çevir
        url_pattern = None
        if url_filter:
            url_pattern = re.compile(url_filter)
        
        # Sayfa sayısını sınırla
        page_count = 0
        
        while url_queue and page_count < max_pages:
            # Kuyruktaki bir sonraki URL'i al
            current_url, current_depth = url_queue.pop(0)
            
            # Zaten ziyaret edilmiş mi kontrol et
            if current_url in visited_urls:
                continue
            
            # Ziyaret edildi olarak işaretle
            visited_urls.add(current_url)
            page_count += 1
            
            self.logger.info(f"[{page_count}/{max_pages}] [{current_depth}/{max_depth}] Ziyaret ediliyor: {current_url}")
            
            try:
                # Sayfayı indir
                response = requests.get(current_url, verify=False, timeout=30)
                if response.status_code != 200:
                    self.logger.warning(f"  Hata: HTTP {response.status_code}")
                    continue
                
                # HTML'i parse et
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Sayfa içeriğini kaydet
                content = soup.get_text()
                results[current_url] = {
                    "content": content,
                    "title": soup.title.string if soup.title else "",
                    "depth": current_depth
                }
                self.logger.info(f"  İçerik alındı: {len(content)} karakter")
                
                # Maksimum derinliğe ulaşıldıysa linkleri takip etme
                if current_depth >= max_depth:
                    continue
                
                # Sayfadaki tüm linkleri bul
                links = soup.find_all("a", href=True)
                self.logger.info(f"  {len(links)} link bulundu")
                
                # Her linki işle
                for link in links:
                    href = link["href"]
                    
                    # Mutlak URL oluştur
                    absolute_url = urljoin(current_url, href)
                    
                    # Fragment'i kaldır (#sonrası)
                    absolute_url = absolute_url.split("#")[0]
                    
                    # Boş URL'leri atla
                    if not absolute_url:
                        continue
                    
                    # Zaten ziyaret edilmiş mi kontrol et
                    if absolute_url in visited_urls:
                        continue
                    
                    # Domain filtresini uygula
                    if domain_filter:
                        parsed_url = urlparse(absolute_url)
                        if parsed_url.netloc != domain_filter:
                            continue
                    
                    # URL filtresini uygula
                    if url_pattern and not url_pattern.search(absolute_url):
                        continue
                    
                    # Kuyruğa ekle
                    url_queue.append((absolute_url, current_depth + 1))
            
            except Exception as e:
                self.logger.error(f"  Hata: {str(e)}")
        
        self.logger.info(f"Toplam {page_count} sayfa ziyaret edildi.")
        return results
    
    def extract_links_from_page(self, url: str, domain_filter: str = None) -> List[Dict[str, str]]:
        """
        Belirli bir sayfadaki tüm linkleri çıkarır.
        
        Args:
            url: İncelenecek URL
            domain_filter: Sadece bu domain içindeki linkleri döndür (None: tüm domainler)
            
        Returns:
            list: Bulunan linkler listesi
        """
        from urllib.parse import urlparse, urljoin
        
        links = []
        
        try:
            # Sayfayı indir
            response = requests.get(url, verify=False, timeout=30)
            if response.status_code != 200:
                self.logger.warning(f"Hata: HTTP {response.status_code}")
                return links
            
            # HTML'i parse et
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Tüm linkleri bul
            a_tags = soup.find_all("a", href=True)
            
            # Domain filtresini ayarla
            if domain_filter is None:
                parsed_url = urlparse(url)
                domain_filter = parsed_url.netloc
            
            # Her linki işle
            for a_tag in a_tags:
                href = a_tag["href"]
                
                # Mutlak URL oluştur
                absolute_url = urljoin(url, href)
                
                # Fragment'i kaldır (#sonrası)
                absolute_url = absolute_url.split("#")[0]
                
                # Boş URL'leri atla
                if not absolute_url:
                    continue
                
                # Domain filtresini uygula
                if domain_filter:
                    parsed_url = urlparse(absolute_url)
                    if parsed_url.netloc != domain_filter:
                        continue
                
                # Link metnini al
                link_text = a_tag.get_text().strip()
                
                links.append({
                    "url": absolute_url,
                    "text": link_text
                })
            
            self.logger.info(f"{url} adresinde {len(links)} link bulundu.")
            
        except Exception as e:
            self.logger.error(f"Linkler çıkarılırken hata: {str(e)}")
        
        return links
        
    def process_urls(self) -> List[Document]:
        """
        URL listesindeki tüm web sayfalarını işler ve Document listesi olarak döndürür.
        
        Returns:
            List[Document]: İşlenen dokümanlar listesi
        """
        documents = []
        
        if not self.urls:
            self.logger.warning("URL listesi boş, web içeriği işlenemedi")
            return documents
        
        for i, url in enumerate(self.urls):
            try:
                self.logger.info(f"URL işleniyor ({i+1}/{len(self.urls)}): {url}")
                content = self.scrape_url(url)
                
                if not content:
                    self.logger.warning(f"URL içeriği çekilemedi: {url}")
                    continue
                
                # Document oluştur
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "type": "web",
                        "title": url,
                        "processed_date": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                self.logger.info(f"URL başarıyla işlendi: {url} - İçerik uzunluğu: {len(content)} karakter")
            except Exception as e:
                self.logger.error(f"URL işlenirken hata: {url} - {str(e)}")
        
        return documents
    

    
    def process_with_link_following(self, start_urls: List[str], max_depth: int = 2, 
                                   max_pages_per_url: int = 20) -> List[Document]:
        """
        Başlangıç URL'lerinden linkleri takip ederek veri çeker ve Document listesi olarak döndürür.
        
        Args:
            start_urls: Başlangıç URL'leri listesi
            max_depth: Maksimum takip derinliği
            max_pages_per_url: Her başlangıç URL'i için maksimum sayfa sayısı
            
        Returns:
            List[Document]: İşlenen dokümanlar listesi
        """
        documents = []
        
        for start_url in start_urls:
            results = self.scrape_with_link_following(
                start_url=start_url,
                max_depth=max_depth,
                max_pages=max_pages_per_url
            )
            
            for url, data in results.items():
                documents.append(Document(
                    page_content=data["content"],
                    metadata={
                        "source": url,
                        "title": data["title"],
                        "depth": data["depth"]
                    }
                ))
        
        return documents
    
    def process_paginated_sites(self, site_configs: List[Dict[str, Any]], 
                               use_selenium: bool = False) -> List[Document]:
        """
        Çok sayfalı siteleri işler ve Document listesi olarak döndürür.
        
        Args:
            site_configs: Site yapılandırmaları listesi
            use_selenium: Selenium kullanılsın mı?
            
        Returns:
            List[Document]: İşlenen dokümanlar listesi
        """
        documents = []
        
        for site_config in site_configs:
            self.logger.info(f"Çok sayfalı site işleniyor: {site_config['base_url']}")
            
            # Selenium veya Requests kullanarak veri çek
            if use_selenium:
                contents = self.scrape_with_selenium(site_config)
            else:
                contents = self.scrape_paginated_site_with_requests(site_config)
            
            # Her sayfadan çekilen içeriği işle
            for i, content in enumerate(contents):
                if content:
                    page_num = site_config["start"] + i
                    url = f"{site_config['base_url']}?{site_config['param']}={page_num}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "title": f"Sayfa {page_num}",
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "type": "web"
                        }
                    )
                    documents.append(doc)
        
        return documents
        
    def scrape_structured_content(self, url: str, config: Dict[str, Any], use_selenium: bool = False) -> Dict[str, Any]:
        """
        Yapılandırılmış içeriği çeker. Bu fonksiyon, farklı türdeki web sitelerinden veri çekmek için 
        esnek bir çözüm sunar. Soru-cevap sayfaları, tablolar, listeler, vb. için kullanılabilir.
        
        Args:
            url: İçeriği çekilecek URL
            config: Veri çekme yapılandırması. Şu anahtarları içerebilir:
                - selectors: CSS seçicileri sözlüğü (key: veri adı, value: CSS seçici)
                - paired_selectors: Eşleştirilmiş seçiciler (key: grup adı, value: {item1_selector, item2_selector})
                - click_selectors: Tıklanacak elementlerin seçicileri (Selenium ile kullanılır)
                - wait_selectors: Bekleme yapılacak elementlerin seçicileri (Selenium ile kullanılır)
                - data_type: Veri tipi ("faq", "table", "list", vb.)
                - data_format: Çıktı formatı ("json", "csv", "text")
            use_selenium: Selenium kullanılsın mı? Dinamik içerik için True olarak ayarlayın.
            
        Returns:
            Dict[str, Any]: Çekilen yapılandırılmış veri
        """
        self.logger.info(f"Yapılandırılmış içerik çekiliyor: {url}")
        
        # Önce Selenium kullanımını kontrol et
        if use_selenium or config.get("force_selenium", False):
            return self._scrape_with_selenium(url, config)
        
        # Standart requests ile deneyelim
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'tr,en-US;q=0.7,en;q=0.3'
            }
            
            response = requests.get(url, headers=headers, verify=False, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                return self._process_structured_content(soup, config)
            else:
                self.logger.warning(f"URL çekilemedi ({url}): HTTP {response.status_code}")
                # Selenium ile deneyelim
                return self._scrape_with_selenium(url, config)
                
        except Exception as e:
            self.logger.warning(f"Yapılandırılmış içerik çekilirken hata: {str(e)}")
            # Hata durumunda Selenium ile deneyelim
            return self._scrape_with_selenium(url, config)
    
    def _process_structured_content(self, soup: BeautifulSoup, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        BeautifulSoup nesnesi üzerinden yapılandırılmış içeriği işler.
        
        Args:
            soup: BeautifulSoup nesnesi
            config: Veri çekme yapılandırması
            
        Returns:
            Dict[str, Any]: Çekilen yapılandırılmış veri
        """
        result = {"url": config.get("url", ""), "data_type": config.get("data_type", "unknown")}
        data_type = config.get("data_type", "unknown")
        
        # Sayfa başlığını al (varsa)
        title_selectors = ["h1", "h1.page-title", ".page-header h1", ".entry-title", "title"]
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                result["title"] = title_elem.get_text(strip=True)
                break
        
        # İçerik türüne göre işleme yap
        # 1. Soru-Cevap (FAQ) içeriği
        if data_type == "faq":
            question_selector = config.get("question_selector", "h2, h3, h4, dt, .question, .faq-question")
            answer_selector = config.get("answer_selector", "p, dd, .answer, .faq-answer")
            
            questions = soup.select(question_selector)
            answers = soup.select(answer_selector)
            
            faq_items = []
            
            # Eşit sayıda soru ve cevap varsa doğrudan eşleştir
            if len(questions) == len(answers):
                for i in range(len(questions)):
                    faq_items.append({
                        "question": questions[i].get_text(strip=True),
                        "answer": answers[i].get_text(strip=True)
                    })
            # Soru ve cevapları komşuluk ilişkisine göre eşleştir
            else:
                for question in questions:
                    answer_text = ""
                    next_elem = question.find_next_sibling()
                    
                    # Bir sonraki soru veya başlığa kadar olan tüm içeriği topla
                    while next_elem and not next_elem.name in ["h2", "h3", "h4", "dt"] and not next_elem.select_one(question_selector):
                        if next_elem.name in ["p", "div", "span", "dd"] or next_elem.select_one(answer_selector):
                            answer_text += next_elem.get_text(strip=True) + "\n"
                        next_elem = next_elem.find_next_sibling()
                    
                    if answer_text:
                        faq_items.append({
                            "question": question.get_text(strip=True),
                            "answer": answer_text.strip()
                        })
            
            result["faq_items"] = faq_items
        
        # 2. Tablo içeriği
        elif data_type == "table":
            table_selector = config.get("table_selector", "table")
            tables = soup.select(table_selector)
            
            result["tables"] = []
            
            for table in tables:
                rows = table.select("tr")
                if not rows:
                    continue
                    
                # Başlıkları al
                headers = []
                header_row = rows[0].select("th")
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row]
                    rows = rows[1:]  # Başlık satırını atla
                
                # Tablo verilerini al
                table_data = []
                for row in rows:
                    cells = row.select("td")
                    if cells:
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        table_data.append(row_data)
                
                if table_data:
                    result["tables"].append(table_data)
                    
                if headers and not "headers" in result:
                    result["headers"] = headers
        
        # 3. Liste içeriği
        elif data_type == "list":
            list_selector = config.get("list_selector", "ul li, ol li, .list-item")
            list_items = soup.select(list_selector)
            
            result["list_items"] = [item.get_text(strip=True) for item in list_items]
        
        # 4. Eşleştirilmiş içerik (başlık-açıklama gibi)
        elif data_type == "paired":
            paired_selectors = config.get("paired_selectors", {})
            result["paired_content"] = {}
            
            for group_name, selectors in paired_selectors.items():
                item1_selector = selectors.get("item1_selector")
                item2_selector = selectors.get("item2_selector")
                format_type = selectors.get("format", "array")
                
                if item1_selector and item2_selector:
                    items1 = soup.select(item1_selector)
                    items2 = soup.select(item2_selector)
                    
                    # Format: key-value
                    if format_type == "kv" and len(items1) == len(items2):
                        result["paired_content"][group_name] = {}
                        for i in range(len(items1)):
                            key = items1[i].get_text(strip=True)
                            value = items2[i].get_text(strip=True)
                            result["paired_content"][group_name][key] = value
                    
                    # Format: array of pairs
                    elif format_type == "array" and len(items1) == len(items2):
                        result["paired_content"][group_name] = []
                        for i in range(len(items1)):
                            result["paired_content"][group_name].append({
                                "item1": items1[i].get_text(strip=True),
                                "item2": items2[i].get_text(strip=True)
                            })
                    
                    # Komşuluk ilişkisine göre eşleştirme
                    elif format_type == "adjacent":
                        result["paired_content"][group_name] = []
                        for item1 in items1:
                            item2 = item1.find_next(item2_selector)
                            if item2:
                                result["paired_content"][group_name].append({
                                    "item1": item1.get_text(strip=True),
                                    "item2": item2.get_text(strip=True)
                                })
        
        # 5. Genel içerik
        elif data_type == "content":
            # Tekil seçiciler
            if "selectors" in config:
                for key, selector in config["selectors"].items():
                    elements = soup.select(selector)
                    if elements:
                        if len(elements) == 1:
                            result[key] = elements[0].get_text(strip=True)
                        else:
                            result[key] = [el.get_text(strip=True) for el in elements]
        
        # Eşleştirilmiş seçiciler (soru-cevap, anahtar-değer çiftleri vb.)
        if "paired_selectors" in config:
            for group_name, selectors in config["paired_selectors"].items():
                if len(selectors) >= 2:
                    item1_selector = selectors.get("item1_selector")
                    item2_selector = selectors.get("item2_selector")
                    
                    if item1_selector and item2_selector:
                        item1_elements = soup.select(item1_selector)
                        item2_elements = soup.select(item2_selector)
                        
                        # Eşleştirilmiş elementleri işle
                        pairs = []
                        for i in range(min(len(item1_elements), len(item2_elements))):
                            item1_text = item1_elements[i].get_text(strip=True)
                            item2_text = item2_elements[i].get_text(strip=True)
                            
                            # Çift formatını belirle
                            pair_format = selectors.get("format", "dict")
                            if pair_format == "dict":
                                pairs.append({"item1": item1_text, "item2": item2_text})
                            elif pair_format == "qa":
                                pairs.append({"question": item1_text, "answer": item2_text})
                            elif pair_format == "kv":
                                pairs.append({"key": item1_text, "value": item2_text})
                            else:
                                pairs.append([item1_text, item2_text])
                        
                        result[group_name] = pairs
        
        # Tablo içeriği
        if config.get("data_type") == "table" and "table_selector" in config:
            tables = soup.select(config["table_selector"])
            table_data = []
            
            for table in tables:
                rows = table.select("tr")
                table_rows = []
                
                for row in rows:
                    # Başlık satırı mı kontrol et
                    headers = row.select("th")
                    if headers:
                        header_row = [header.get_text(strip=True) for header in headers]
                        result["headers"] = header_row
                    else:
                        # Normal satır
                        cells = row.select("td")
                        if cells:
                            row_data = [cell.get_text(strip=True) for cell in cells]
                            table_rows.append(row_data)
                
                table_data.append(table_rows)
            
            result["tables"] = table_data
        
        # Liste içeriği
        if config.get("data_type") == "list" and "list_selector" in config:
            list_items = soup.select(config["list_selector"])
            result["list_items"] = [item.get_text(strip=True) for item in list_items]
        
        # Soru-cevap içeriği (FAQ)
        if config.get("data_type") == "faq":
            question_selector = config.get("question_selector")
            answer_selector = config.get("answer_selector")
            
            if question_selector and answer_selector:
                questions = soup.select(question_selector)
                answers = soup.select(answer_selector)
                
                faq_pairs = []
                for i in range(min(len(questions), len(answers))):
                    question_text = questions[i].get_text(strip=True)
                    answer_text = answers[i].get_text(strip=True)
                    
                    faq_pairs.append({
                        "question": question_text,
                        "answer": answer_text
                    })
                
                result["faq_items"] = faq_pairs
        
        return result
    
    def _scrape_with_selenium(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selenium kullanarak yapılandırılmış içeriği çeker.
        Dinamik içerikli web siteleri için uygundur.
        
        Args:
            url: İçeriği çekilecek URL
            config: Veri çekme yapılandırması
            
        Returns:
            Dict[str, Any]: Çekilen yapılandırılmış veri
        """
        self.logger.info(f"Selenium ile yapılandırılmış içerik çekiliyor: {url}")
        
        try:
            # Selenium kurulu değilse uyarı ver
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from webdriver_manager.chrome import ChromeDriverManager
            except ImportError:
                self.logger.warning("Selenium kurulu değil. 'pip install selenium webdriver-manager' komutu ile kurabilirsiniz.")
                return {"error": "Selenium kurulu değil"}
            
            # Chrome seçeneklerini ayarla
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # WebDriver'ı başlat
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            
            # URL'yi ziyaret et
            driver.get(url)
            
            # Sayfanın yüklenmesini bekle
            wait_timeout = config.get("wait_timeout", 10)
            wait = WebDriverWait(driver, wait_timeout)
            
            # Beklenecek elementler varsa bekle
            if "wait_selectors" in config:
                for selector in config["wait_selectors"]:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    except Exception as e:
                        self.logger.warning(f"Bekleme sırasında hata: {str(e)}")
            
            # Tıklanacak elementler varsa tıkla
            if "click_selectors" in config:
                for selector in config["click_selectors"]:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            driver.execute_script("arguments[0].click();", element)
                            # Tıklama sonrası kısa bir bekleme
                            import time
                            time.sleep(config.get("click_delay", 1))
                    except Exception as e:
                        self.logger.warning(f"Tıklama sırasında hata: {str(e)}")
            
            # Sayfa içeriğini al
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            
            # WebDriver'ı kapat
            driver.quit()
            
            # İçeriği işle
            return self._process_structured_content(soup, config)
            
        except Exception as e:
            self.logger.warning(f"Selenium ile içerik çekilirken hata: {str(e)}")
            return {"error": str(e)}
            
    # Geriye uyumluluk için eski fonksiyonlar
    def scrape_faq_content(self, url: str, question_selector: str, answer_selector: str) -> List[Dict[str, str]]:
        """
        Soru-cevap (FAQ) içeriğini çeker. (Geriye uyumluluk için)
        
        Args:
            url: İçeriği çekilecek URL
            question_selector: Soruları seçmek için CSS seçici
            answer_selector: Cevapları seçmek için CSS seçici
            
        Returns:
            List[Dict[str, str]]: Soru-cevap çiftlerinin listesi
        """
        config = {
            "data_type": "faq",
            "question_selector": question_selector,
            "answer_selector": answer_selector
        }
        
        result = self.scrape_structured_content(url, config)
        return result.get("faq_items", [])


# Modül doğrudan çalıştırıldığında örnek kullanım
if __name__ == "__main__":
    # Kadına yönelik şiddet ve destek konularında web sayfaları
    URLLIST = [
        "https://morcati.org.tr/siddete-ugradiginizda-neler-yapabilirsiniz/",
        "https://www.kadindayanismavakfi.org.tr/siddete-maruz-kaldigimizda-ne-yapmaliyiz/",
    ]
    
    # WebScraper örneği oluştur
    scraper = WebScraper(urls=URLLIST)
    
    # URL'leri işle
    documents = scraper.process_urls()
    
    # Sonuçları göster
    for doc in documents:
        print(f"URL: {doc.metadata['source']}")
        print(f"İçerik uzunluğu: {len(doc.page_content)} karakter")
        print("-" * 50)
