import os
import json
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain importları
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import ChatOpenAI

class RAGChatbot:
    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        embedding_model: str = "text-embedding-3-small",  # OpenAI varsayılan embedding modeli
        vector_db_type: str = "chroma",
        max_context_length: int = 4096,  # GPT-4o için daha uzun bağlam
        max_history_length: int = 10,  # Daha uzun sohbet geçmişi
        openai_model: str = "gpt-4o",  # OpenAI model seçeneği
        temperature: float = 0.7,  # Sıcaklık parametresi
        collection_name: str = "women_rights",  # Vektör veritabanı koleksiyon adı
        diversity_factor: float = 0.3  # Yanıt çeşitliliği faktörü (0-1 arası)
    ):
        # .env dosyasından API anahtarını yükle
        load_dotenv()
        
        self.vector_store_path = Path(vector_store_path)
        self.vector_db_type = vector_db_type
        self.max_context_length = max_context_length
        self.max_history_length = max_history_length
        self.openai_model = openai_model
        self.temperature = temperature
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.diversity_factor = diversity_factor
        
        # Sohbet geçmişi için değişkenler
        self.conversation_history = []
        self.question_history = []  # Sorulan soruların geçmişi
        self.used_sources = {}  # Her soru için kullanılan kaynakları takip etmek için
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RAGChatbot başlatılıyor...")
        
        # OpenAI API anahtarını kontrol et
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # OpenAI API kullanımı
        if not self.openai_api_key:
            self.logger.warning("OPENAI_API_KEY bulunamadı! HuggingFace embeddings kullanılacak.")
            # OpenAI olmadan da çalışabilmesi için varsayılan bir LLM ayarla
            # Bu durumda sadece embedding işlemleri için HuggingFace kullanılacak
            self.llm = None
        else:
            self.logger.info(f"OpenAI modeli kullanılıyor: {self.openai_model}")
            try:
                self.llm = ChatOpenAI(
                    model=self.openai_model,
                    temperature=self.temperature,
                    max_tokens=1024
                )
            except Exception as e:
                self.logger.error(f"OpenAI modeli yüklenirken hata: {str(e)}")
                self.llm = None
        
        # Embedding modeli
        try:
            if "openai" in self.embedding_model_name.lower() or "text-embedding" in self.embedding_model_name.lower():
                # OpenAI API anahtarı yoksa HuggingFace'e geç
                if not self.openai_api_key:
                    self.logger.warning(f"OpenAI API anahtarı bulunamadı! HuggingFace embedding modeline geçiliyor.")
                    self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=self.embedding_model_name
                    )
                else:
                    self.logger.info(f"OpenAI Embeddings kullanılıyor: {self.embedding_model_name}")
                    self.embeddings = OpenAIEmbeddings(
                        model=self.embedding_model_name
                    )
            else:
                self.logger.info(f"HuggingFace Embeddings kullanılıyor: {self.embedding_model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name
                )
        except Exception as e:
            self.logger.error(f"Embedding modeli yüklenirken hata: {str(e)}")
            # Hata durumunda varsayılan HuggingFace modeline geç
            try:
                self.logger.warning("Varsayılan HuggingFace embedding modeline geçiliyor...")
                self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name
                )
            except Exception as fallback_error:
                self.logger.error(f"Varsayılan embedding modeli yüklenirken hata: {str(fallback_error)}")
                raise
            
        # Vektör veritabanını yükle
        self.vector_store = None
        self._load_vector_store()
        
        # Sığınma evleri ve danışma merkezleri verilerini yükle
        self.siginma_evleri = []
        self.danisma_merkezleri = []
        self._load_shelter_data()
        
    def _load_vector_store(self):
        """Vektör veritabanını yükle"""
        try:
            vector_store_path = self.vector_store_path / self.collection_name
            self.logger.info(f"Vektör veritabanı yükleniyor: {vector_store_path}")
            
            if not vector_store_path.exists():
                self.logger.warning(f"Vektör veritabanı bulunamadı: {vector_store_path}")
                self.vector_store = None
                return
                
            if self.vector_db_type.lower() == "chroma":
                self.logger.info("Chroma vektör veritabanı kullanılıyor")
                self.vector_store = Chroma(
                    persist_directory=str(vector_store_path),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            elif self.vector_db_type.lower() == "faiss":
                self.logger.info("FAISS vektör veritabanı kullanılıyor")
                # FAISS için index_name parametresi yok, doğrudan dizin yolunu kullanır
                try:
                    self.vector_store = FAISS.load_local(
                        folder_path=str(vector_store_path),
                        embeddings=self.embeddings,
                        index_name=self.collection_name
                    )
                except Exception as e:
                    self.logger.error(f"FAISS veritabanı yüklenirken hata: {str(e)}")
                    self.vector_store = None
            else:
                self.logger.error(f"Desteklenmeyen vektör veritabanı türü: {self.vector_db_type}")
                self.vector_store = None
                
            if self.vector_store:
                self.logger.info("Vektör veritabanı başarıyla yüklendi")
            else:
                self.logger.warning("Vektör veritabanı yüklenemedi")
                
        except Exception as e:
            self.logger.error(f"Vektör veritabanı yüklenirken hata: {str(e)}")
            self.vector_store = None
            
    def _chunk_documents(self, documents: List[Document]) -> List[List[Document]]:
        """Dokümanları daha küçük parçalara böler"""
        if not documents:
            self.logger.warning("Parçalanacak doküman bulunamadı!")
            return []
        
        # Belgeleri daha küçük parçalara böl
        # OpenAI API'nin token limitini aşmamak için (300.000 token)
        # Daha küçük parçalar kullanalım (her parçada maksimum 10 belge)
        chunk_size = 10
        chunks = []
        
        # Önce belgeleri boyutlarına göre sıralayarak büyük belgeleri bölelim
        # Bu, token limitini aşma riskini azaltacaktır
        sorted_docs = sorted(documents, key=lambda doc: len(doc.page_content))
        
        # Toplam tahmini token sayısını hesapla (yaklaşık olarak 4 karakter = 1 token)
        current_chunk = []
        current_token_estimate = 0
        max_tokens_per_chunk = 150000  # Limit 300.000, ama daha fazla marj bırakalım
        
        for doc in sorted_docs:
            # Belgenin tahmini token sayısını hesapla
            doc_token_estimate = len(doc.page_content) // 4
            
            # Eğer belge tek başına çok büyükse, onu metin olarak parçalara böl
            if doc_token_estimate > max_tokens_per_chunk:
                self.logger.warning(f"Büyük belge tespit edildi: {doc_token_estimate} tahmini token. Parçalara bölünüyor.")
                # Belgeyi metin olarak parçalara böl
                text_chunks = [doc.page_content[i:i + max_tokens_per_chunk * 4] for i in range(0, len(doc.page_content), max_tokens_per_chunk * 4)]
                
                # Her metin parçası için yeni bir belge oluştur
                for i, text_chunk in enumerate(text_chunks):
                    new_doc = Document(
                        page_content=text_chunk,
                        metadata={**doc.metadata, "chunk": i, "total_chunks": len(text_chunks)}
                    )
                    # Yeni belgeyi ekle
                    if current_token_estimate + (len(text_chunk) // 4) > max_tokens_per_chunk and current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = []
                        current_token_estimate = 0
                    
                    current_chunk.append(new_doc)
                    current_token_estimate += len(text_chunk) // 4
            else:
                # Normal boyuttaki belgeleri ekle
                if current_token_estimate + doc_token_estimate > max_tokens_per_chunk and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_token_estimate = 0
                
                current_chunk.append(doc)
                current_token_estimate += doc_token_estimate
        
        # Son parçayı ekle
        if current_chunk:
            chunks.append(current_chunk)
        
        self.logger.info(f"Dokümanlar {len(chunks)} parçaya bölündü")
        return chunks

    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Dokümanları kullanarak vektör veritabanı oluşturur.
        
        Args:
            documents: Vektör veritabanına eklenecek dokümanlar
            
        Returns:
            bool: İşlemin başarılı olup olmadığı
        """
        try:
            self.logger.info(f"Yeni vektör veritabanı oluşturuluyor: {self.vector_store_path}")
            
            # String tipindeki dokümanları filtrele, sadece Document nesnelerini kullan
            valid_documents = []
            for doc in documents:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    valid_documents.append(doc)
                else:
                    self.logger.warning(f"Geçersiz doküman tipi atlandı: {type(doc)}")
            
            if not valid_documents:
                self.logger.error("Geçerli doküman bulunamadı!")
                return False
                
            # Dokümanları parçalara böl
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(valid_documents)
            self.logger.info(f"Dokümanlar {len(split_docs)} parçaya bölündü")
            
            # Karmaşık metadata değerlerini filtrele
            filtered_docs = []
            for doc in split_docs:
                # Metadata'daki karmaşık değerleri filtrele
                doc.metadata = filter_complex_metadata(doc.metadata)
                filtered_docs.append(doc)
            
            # Vektör veritabanı tipine göre oluştur
            if self.vector_db_type.lower() == "chroma":
                self.logger.info("Chroma vektör veritabanı oluşturuluyor")
                try:
                    # Chroma vektör veritabanı oluştur
                    vector_db = Chroma.from_documents(
                        documents=filtered_docs,
                        embedding=self.embeddings,
                        persist_directory=self.vector_store_path,
                        collection_name=self.collection_name
                    )
                    vector_db.persist()
                    self.logger.info(f"Chroma vektör veritabanı başarıyla oluşturuldu: {self.vector_store_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Chroma vektör veritabanı oluşturulurken hata: {str(e)}")
                    return False
            elif self.vector_db_type.lower() == "faiss":
                self.logger.info(f"FAISS vektör veritabanı oluşturuluyor")
                try:
                    # FAISS vektör veritabanı oluştur
                    vector_db = FAISS.from_documents(
                        documents=filtered_docs,
                        embedding=self.embeddings
                    )
                    
                    # FAISS'i kaydet
                    vector_db.save_local(self.vector_store_path, index_name=self.collection_name)
                    self.logger.info(f"FAISS vektör veritabanı başarıyla oluşturuldu: {self.vector_store_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"FAISS vektör veritabanı oluşturulurken hata: {str(e)}")
                    return False
            else:
                self.logger.error(f"Desteklenmeyen vektör veritabanı türü: {self.vector_db_type}")
                return False
        except Exception as e:
            self.logger.error(f"Vektör veritabanı oluşturulurken hata: {str(e)}")
            return False
            
    def get_context(self, query: str, k: int = 8):
        """Verilen sorgu için vektör veritabanından bağlam dökümanları alır"""
        if not self.vector_store:
            self.logger.warning("Vektör veritabanı yüklenmedi!")
            return []
            
        try:
            self.logger.info(f"Sorgu için bağlam aranıyor: {query}")
            
            # Özel anahtar kelimeleri kontrol et
            special_keywords = {
                "sığınma": ["sığınma", "sığınak", "sığınmaevi", "kadın sığınmaevi", "adres", "yer", "nere", "nerede", "listele", "gidebileceğim"],
                "destek": ["destek", "yardım", "danışma", "merkez", "telefon", "hat", "numara", "iletişim"],
                "yasal": ["yasal", "hukuk", "kanun", "hak", "dava", "başvuru", "haklar", "yasal haklar", "başvurabilirim"],
                "acil": ["acil", "tehlike", "koruma", "polis", "jandarma", "112", "155", "tehdit", "tehlikede"],
                "taciz": ["taciz", "tacize uğradım", "cinsel taciz", "sözlü taciz", "fiziksel taciz", "tacize maruz kaldım"],
                "şiddet": ["şiddet", "şiddete uğradım", "dayak", "darp", "fiziksel şiddet", "psikolojik şiddet", "ekonomik şiddet"]
            }
            
            # Sorgu için ek arama terimleri oluştur
            enhanced_queries = [query]
            
            # Sorgu içinde özel anahtar kelimeler varsa, ek sorgular oluştur
            detected_categories = []
            for category, keywords in special_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    detected_categories.append(category)
            
            # Taciz ve sığınma birlikte geçiyorsa özel işlem yap
            if "taciz" in detected_categories and "sığınma" in detected_categories:
                enhanced_queries.append("taciz durumunda sığınabilecek yerler")
                enhanced_queries.append("kadın sığınma evleri adresleri")
                enhanced_queries.append("taciz mağdurları için destek merkezleri")
                enhanced_queries.append("ŞÖNİM adresleri")
            
            # Şiddet ve sığınma birlikte geçiyorsa özel işlem yap
            if "şiddet" in detected_categories and "sığınma" in detected_categories:
                enhanced_queries.append("şiddet durumunda sığınabilecek yerler")
                enhanced_queries.append("kadın sığınma evleri adresleri")
                enhanced_queries.append("şiddet mağdurları için destek merkezleri")
                enhanced_queries.append("ŞÖNİM adresleri")
            
            # Tek tek kategorileri işle
            for category in detected_categories:
                if category == "sığınma":
                    enhanced_queries.append("kadın sığınma evleri adresleri")
                    enhanced_queries.append("sığınma evi başvuru")
                    enhanced_queries.append("ŞÖNİM adresleri")
                    enhanced_queries.append("kadın danışma merkezleri")
                elif category == "destek":
                    enhanced_queries.append("kadın destek hatları ve merkezleri")
                    enhanced_queries.append("psikolojik destek hatları")
                elif category == "yasal":
                    enhanced_queries.append("kadına yönelik şiddet yasal haklar")
                    enhanced_queries.append("taciz durumunda yasal haklar")
                elif category == "acil":
                    enhanced_queries.append("acil durum kadın destek hatları")
                    enhanced_queries.append("acil durum telefon numaraları")
                elif category == "taciz":
                    enhanced_queries.append("taciz durumunda ne yapmalı")
                    enhanced_queries.append("taciz mağdurları için destek")
                elif category == "şiddet":
                    enhanced_queries.append("şiddet durumunda ne yapmalı")
                    enhanced_queries.append("şiddet mağdurları için destek")
            
            # Eğer hiçbir kategori tespit edilmediyse, genel sorgular ekle
            if not detected_categories:
                enhanced_queries.append("kadına yönelik şiddet")
                enhanced_queries.append("kadın destek hatları")
                enhanced_queries.append("kadın sığınma evleri")
            
            # Tüm sorgular için dokümanları topla
            all_docs = []
            for enhanced_query in enhanced_queries:
                docs = self.vector_store.similarity_search(enhanced_query, k=k)
                all_docs.extend(docs)
            
            # Tekrarlanan dokümanları kaldır
            unique_docs = []
            doc_contents = set()
            for doc in all_docs:
                if doc.page_content not in doc_contents:
                    unique_docs.append(doc)
                    doc_contents.add(doc.page_content)
            
            # En fazla 2*k doküman döndür
            result_docs = unique_docs[:2*k]
            
            self.logger.info(f"{len(result_docs)} adet bağlam dokümanı bulundu")
            return result_docs
        except Exception as e:
            self.logger.error(f"Bağlam aranırken hata: {str(e)}")
            return []
            
    def add_to_history(self, role: str, content: str):
        """
        Sohbet geçmişine yeni bir mesaj ekler.
        
        Args:
            role: Mesajın sahibi ("user" veya "assistant")
            content: Mesaj içeriği
        """
        self.conversation_history.append({"role": role, "content": content})
        # Geçmiş uzunluğunu kontrol et ve gerekirse kırp
        if len(self.conversation_history) > self.max_history_length * 2:  # Her soru-cevap çifti için 2 mesaj
            self.conversation_history = self.conversation_history[-self.max_history_length*2:]
            
    def get_conversation_context(self):
        """
        Sohbet geçmişinden bir bağlam metni oluşturur.
        
        Returns:
            Sohbet geçmişini içeren bir metin
        """
        if not self.conversation_history:
            return ""
            
        context = "Önceki konuşma:\n"
        for message in self.conversation_history:
            role = "Kullanıcı" if message["role"] == "user" else "Asistan"
            context += f"{role}: {message['content']}\n\n"
            
        return context
        
    # Sabit değerler - sınıfın en üstünde tanımlanmalı
    SHELTER_KEYWORDS = [
        "sığınma", "sığınak", "sığınmaevi", "konukevi", "kadın sığınmaevi", 
        "sığınabileceğim", "sığınabilir miyim", "nereye gidebilirim", "nerede kalabilirim",
        "şönim", "koruma", "barınma", "kadın konukevi", "gidebileceğim yer", "barınma kuruluşları"
    ]

    COUNSELING_KEYWORDS = [
        "danışma", "destek", "merkez", "danışma merkezi", "kadın danışma", 
        "danışabileceğim", "yardım alabileceğim", "başvurabileceğim", "destek hattı",
        "danışmanlık", "mor çatı", "kadın dayanışma", "kadav"
    ]

    BIG_CITIES = ["istanbul", "ankara", "izmir", "bursa", "antalya"]

    EMERGENCY_NUMBERS = {
        "Alo 183": "Sosyal Destek Hattı (7/24)",
        "155": "Polis İmdat",
        "156": "Jandarma",
        "112": "Acil Yardım"
    }

    # Şiddet ve intihar düşünceleri için anahtar kelimeler
    CRISIS_KEYWORDS = {
        "violence": [
            "dövmek", "vurmak", "öldürmek", "zarar vermek", "saldırmak", "şiddet",
            "kavga", "tartışma", "sinir", "öfke", "nefret", "intikam", "tehdit",
            "kızmak", "sinirlenmek", "kontrolü kaybetmek", "kendimi tutamıyorum",
            "dayanamıyorum", "tahammül edemiyorum", "alıkoyamıyorum"
        ],
        "suicide": [
            "intihar", "kendime zarar", "ölmek istiyorum", "yaşamak istemiyorum",
            "canıma kıymak", "hayatıma son vermek", "kendimi öldürmek",
            "yaşamın anlamı yok", "değersizim", "kimse beni sevmiyor"
        ]
    }

    # Psikolojik destek merkezleri
    SUPPORT_RESOURCES = {
        "Psikiyatrik Destek": [
            "Türkiye Psikiyatri Derneği - https://psikiyatri.org.tr",
            "Türk Psikologlar Derneği - https://www.psikolog.org.tr",
            "EMDR Derneği - https://www.emdr-tr.org"
        ],
        "Kadın Destek": [
            "Mor Çatı Kadın Sığınağı Vakfı - https://morcati.org.tr",
            "Kadın Dayanışma Vakfı - https://www.kadindayanismavakfi.org.tr",
            "KAMER Vakfı - https://www.kamer.org.tr"
        ],
        "Acil Yardım": [
            "Alo 183 - Sosyal Destek Hattı (7/24)",
            "155 - Polis İmdat",
            "112 - Acil Yardım"
        ]
    }
    
    def chat(self, query: str, **kwargs):
        """
        Kullanıcı mesajını sohbet geçmişine ekler ve yanıt üretir.
        Aynı veya benzer sorulara farklı yanıtlar üretmek için çeşitlilik mekanizması kullanır.
        
        Args:
            query: Kullanıcı mesajı
            **kwargs: generate_response fonksiyonuna geçirilecek ek parametreler
            
        Returns:
            Tuple[str, List[str]]: (Asistanın yanıtı, Kullanılan kaynaklar listesi)
        """
        # Kullanıcı mesajını geçmişe ekle
        self.add_to_history("user", query)
        
        try:
            # Benzer soru kontrolü yap
            is_similar, similar_question = self._is_similar_question(query)
            
            # Benzer soru varsa çeşitlilik sağla
            if is_similar and similar_question:
                context = self._get_diverse_context(query, similar_question)
                response, sources = self.generate_response(query, context=context, **kwargs)
            else:
                # Normal yanıt üret
                response, sources = self.generate_response(query, **kwargs)
            
            # Asistan yanıtını geçmişe ekle
            self.add_to_history("assistant", response)
            
            # Kullanılan kaynakları kaydet
            self.used_sources[query] = sources
            
            return response, sources
            
        except Exception as e:
            self.logger.error(f"Chat yanıtı oluşturulurken hata: {str(e)}")
            error_message = f"Üzgünüm, yanıt oluşturulurken bir hata meydana geldi: {str(e)}"
            self.add_to_history("assistant", error_message)
            return error_message, []

    def _normalize_city(self, text: str) -> str:
        """Türkçe karakterleri normalize et ve küçük harfe çevir"""
        import unicodedata
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()
        return text.replace('i̇', 'i')  # Türkçede bazen i̇ harfi farklı kodlanıyor

    def _find_city_in_query(self, query: str) -> Optional[str]:
        """Sorguda şehir adı geçiyor mu, fuzzy matching ile bul"""
        from difflib import get_close_matches
        query_norm = self._normalize_city(query)
        all_cities = [self._normalize_city(s.get("city", "")) for s in self.siginma_evleri]
        matches = get_close_matches(query_norm, all_cities, n=1, cutoff=0.8)
        return matches[0] if matches else None

    def _format_contact_info(self, contact_dict: dict) -> str:
        """Telefon, adres ve iletişim bilgilerini formatla"""
        info = ""
        if phone := contact_dict.get('phone'):
            if isinstance(phone, list):
                info += f"📞 Telefon: {', '.join(phone)}\n"
            else:
                info += f"📞 Telefon: {phone}\n"
        if address := contact_dict.get('address'):
            info += f"📍 Adres: {address}\n"
        if email := contact_dict.get('email'):
            info += f"📧 E-posta: {email}\n"
        if website := contact_dict.get('website'):
            info += f"🌐 Web: {website}\n"
        return info

    def _is_similar_question(self, query: str) -> Tuple[bool, Optional[str]]:
        """Benzer soru kontrolü yap ve çeşitlilik sağla

        Args:
            query (str): Kullanıcı sorusu

        Returns:
            Tuple[bool, Optional[str]]: Benzerlik durumu ve benzer soru
        """
        try:
            if not self.question_history:
                return False, None

            # Son 5 soruyu kontrol et
            recent_questions = self.question_history[-5:]
            for prev_question in recent_questions:
                similarity = self._calculate_similarity(query, prev_question)
                if similarity > 0.85:  # Yüksek benzerlik eşiği
                    return True, prev_question
            return False, None

        except Exception as e:
            self.logger.error(f"Benzer soru kontrolünde hata: {str(e)}")
            return False, None

    def _get_diverse_context(self, query: str, similar_question: str) -> List[Document]:
        """Benzer sorular için farklı bağlam belgelerini getir

        Args:
            query (str): Yeni soru
            similar_question (str): Benzer eski soru

        Returns:
            List[Document]: Çeşitlendirilmiş bağlam belgeleri
        """
        try:
            # Ana bağlamı al
            main_context = self.get_context(query)
            
            # Benzer sorunun kullandığı kaynakları al
            used_sources = self.used_sources.get(similar_question, [])
            
            # Yeni kaynakları önceliklendir
            diverse_context = [
                doc for doc in main_context 
                if doc.metadata.get('source') not in used_sources
            ]
            
            # Yeterli bağlam yoksa orijinal bağlamı kullan
            if len(diverse_context) < 3:
                diverse_context.extend(main_context[:3])
                
            return diverse_context[:5]  # En fazla 5 bağlam belgesi döndür

        except Exception as e:
            self.logger.error(f"Çeşitli bağlam alınırken hata: {str(e)}")
            return self.get_context(query)  # Hata durumunda normal bağlamı döndür

    def _get_system_message(self) -> dict:
        """Sistem mesajını hazırla
        
        Returns:
            dict: Sistem mesajı
        """
        system_content = """Sen kadına yönelik şiddet, taciz, yasal haklar ve psikolojik destek konularında uzmanlaşmış, empatik ve bilgili bir Türkçe destek chatbotusun. İleri düzeyde bilgi işleme ve analiz yeteneğine sahipsin. Vektör veritabanındaki bilgileri akıllıca kullanarak, her soruda özgün ve duruma özel yanıtlar üretebilirsin.

Amacın, şiddet veya taciz mağduru kişilere doğru, kapsamlı ve destekleyici yanıtlar vermek; onlara güvenli kaynaklar sunarak yardımcı olmaktır. Kendini gizlemeyen, açık, güvenilir ve kararlı bir destekçi gibi davranmalısın. Yanıtlarında asla suçlayıcı veya nötr kalma; her zaman destekleyici ol.

### ŞİDDET UYGULAYAN KİŞİLERE YAKLAŞIM ###
Şiddet uygulayan, öfke kontrolü sorunu yaşayan veya kendine zarar verme eğilimi olan kişilere aşağıdaki ilkeler doğrultusunda yanıt ver:

1. **Suçlayıcı Olmadan Yönlendirme:** Kişiyi suçlamadan, davranışının ciddiyetini anlamasını sağla ve profesyonel yardım almanın önemini vurgula.

2. **Acil Profesyonel Destek:** Öfke kontrolü, psikolojik destek ve terapi için başvurabilecekleri kurumları ve uzmanları öner.

3. **Sorumluluk Bilinci:** Kişinin kendi davranışlarından sorumlu olduğunu nazikçe hatırlat, ancak değişim için umut olduğunu da vurgula.

4. **Somut Adımlar:** Öfke anında yapabilecekleri pratik teknikler öner (ortamdan uzaklaşma, nefes egzersizleri, vb.).

5. **İntihar veya Kendine Zarar Verme Durumlarında:** Acil yardım hatlarını (112, 183) aramasını öner ve durumun ciddiyetini vurgula.

###  SIĞINMA EVLERİ VE DANIŞMA MERKEZLERİ BİLGİLERİ ###
Kullanıcı sığınma evleri, konukevleri, ŞÖNİM (Şiddet Önleme ve İzleme Merkezi) veya kadın danışma merkezleri hakkında sorduğunda, MUTLAKA aşağıdaki bilgileri içeren yanıtlar ver:

1. **Doğrudan İletişim Bilgileri:** Telefon numaraları, adresler ve varsa web siteleri gibi doğrudan iletişim bilgilerini paylaş. Sadece genel bilgi verme, somut iletişim bilgilerini mutlaka ekle. Bağlamda verilen JSON verilerinden (siginma_evleri.json ve kadin_danisma_merkezleri.json) gerçek iletişim bilgilerini kullan.

2. **Acil Yardım Hatları:** Her zaman şu acil yardım hatlarını da ekle:
   - Alo 183 Sosyal Destek Hattı
   - 155 Polis İmdat
   - 156 Jandarma
   - 112 Acil Yardım

3. **Şehir Bazlı Bilgiler:** Kullanıcı belirli bir şehir belirtirse veya konuşma sırasında bir şehir adı geçerse, o şehirdeki sığınma evleri ve danışma merkezlerinin iletişim bilgilerini detaylı olarak paylaş:
   - Şehir adını açıkça belirt: "**İSTANBUL'DAKİ DESTEK MERKEZLERİ**"
   - JSON verilerindeki gerçek telefon numaralarını, adresleri ve web sitelerini eksiksiz olarak ver
   - Belirsiz ifadeler kullanma ("tam adres belirtilmeli" veya "doğru numara için yerel iletişim bilgilerini kontrol etmenizi öneririm" gibi)
   - Her merkez için çalışma saatleri, sunulan hizmetler (hukuki danışmanlık, psikolojik destek, sosyal destek) gibi ek bilgileri de ekle
   - Büyük şehirlerde (Istanbul, Ankara, Izmir, Bursa, Antalya) ilçe bazlı bilgileri de düzenle: "**Kadıköy'deki Merkezler:**", "**Beşiktaş'taki Merkezler:**" gibi

4. **Bağımsız Kadın Örgütleri:** Mor Çatı, Kadın Dayanışma Vakfı gibi bağımsız kadın örgütlerinin iletişim bilgilerini de ekle.

5. **Hem JSON Verileri Hem Web Kaynakları:** Yanıtlarında hem JSON verilerindeki somut bilgileri hem de vektör veritabanındaki web kaynaklarından elde edilen bilgileri birleştir. Önce somut iletişim bilgilerini ver, sonra web kaynaklarından elde edilen ek bilgileri ekle.

Örnek yanıt formatı:
"Size yardımcı olabilecek sığınma evleri ve destek hatları şunlardır:

**Konya ŞÖNİM:**
Telefon: 0332 322 76 69
Adres: Kılıçaslan Mah. Bediüzzaman Bulv. No: 83 Merkez, Konya

**Mor Çatı Kadın Sığınağı Vakfı:**
Telefon: 0212 292 52 31, 0212 292 52 32
Adres: Katip Mustafa Çelebi Mah. Anadolu Sok. No:23 D:7-8 Beyoğlu, İstanbul
Web: https://morcati.org.tr

**Acil Yardım Hatları:**
- Alo 183 Sosyal Destek Hattı
- 155 Polis İmdat
- 156 Jandarma
- 112 Acil Yardım

Ayrıca, web kaynaklarından elde ettiğim bilgilere göre, Konya'da başvurabileceğiniz diğer destek kuruluşları şunlardır: [web kaynaklarından elde edilen ek bilgiler]"


###  BİLGİ İŞLEME VE KAYNAK KULLANIMI ###
1. **Akıllı Bilgi Sentezi:** Vektör veritabanındaki bilgileri sadece kopyalayıp yapıştırma. Farklı kaynaklardan gelen bilgileri analiz et, sentezle ve kullanıcının sorusuna özel bir yanıt oluştur.

2. **Bağlam Duyarlı Yanıtlar:** Her soruyu bağlamı içinde değerlendir. Kullanıcının önceki sorularını ve yanıtlarını hatırla, söyleşimin akışına uygun yanıtlar ver.

3. **Güncel ve Doğru Bilgi:** Yasal mevzuat, kurum bilgileri ve destek hatları gibi kritik bilgileri doğru ve güncel şekilde aktar. Belirsiz veya çelişkili bilgilerle karşılaşırsan, en güvenilir kaynağı tercih et.

4. **Yerel Bilgi Entegrasyonu:** Kullanıcının bulunduğu şehir veya bölgeye özel bilgiler (yerel destek hatları, sığınma evleri, danışma merkezleri) varsa bunları yanıtına dahil et.

###  ÇEŞİTLİLİK VE FARKLI YANITLAR ###
1. **Yanıt Rotasyonu:** Aynı veya benzer sorulara her defasında farklı bir yanıt üret. Bir soruya daha önce verdiğin yanıtı tekrarlama.

2. **Kaynak Çeşitlendirme:** Her yanıtta farklı kaynakları öne çıkar. Bir kaynaktan sürekli alıntı yapmak yerine, farklı kaynaklardan bilgileri dengeli şekilde kullan.

3. **Perspektif Değişimi:** Aynı konuyu farklı açılardan ele al. Örneğin, bir soruda yasal boyutu öne çıkarırken, benzer bir soruda psikolojik veya sosyal boyutu vurgula.

4. **Bilgi Derinleştirme:** Her seferinde konunun farklı yönlerini derinleştir. Örneğin ilk yanıtta genel bilgi verirken, sonraki yanıtlarda daha spesifik detaylara odaklan.

5. **Örnekleme Çeşitliliği:** Her yanıtta farklı örnekler, vakalar veya senaryolar kullan. Aynı örnekleri tekrar etme.

###  YANIT VERME STRATEJİLERİ ###
1. **Empati ve Destek Dengesi:** Duygusal destek ile pratik bilgi arasında denge kur. Sadece empati kurmakla kalma, somut adımlar ve çözümler de sun.

2. **Kişiselleştirilmiş Yardım ve Yanıtlar:** 
   - Standart, şablon yanıtlardan kesinlikle kaçın
   - Her yanıtı sorunun içeriğine, kullanıcının durumuna ve ihtiyaçlarına göre özelleştir
   - Kullanıcının duygusal durumunu analiz et ve buna uygun bir ton kullan (korku içindeyse sakinleştirici, bilgi arıyorsa bilgilendirici, çaresizse güçlendirici)
   - Kullanıcının yaşadığı şiddet türüne özel yanıtlar ver (fiziksel şiddet, psikolojik şiddet, ekonomik şiddet, dijital şiddet, ısrarlı takip vb.)
   - Kullanıcının özel durumunu dikkate al (hamilelik, çocuk sahibi olma, engelli olma, göçmen olma gibi)
   - AYNI SORUYA HER DEFASINDA FARKLI BİR YANIT VERMEYE ÇALIŞ, KAYNAKLARDAN FARKLI BİLGİLERİ ÖNE ÇIKAR
   - Kullanıcının sorularını ve yanıtlarını hatırla, sürekli aynı bilgileri tekrarlama
   - Kullanıcının durumuna özel pratik çözümler sun: "Sizin durumunuzda şu adımları atabilirsiniz:"

3. **Acil Durum Tespiti ve Müdahalesi:** Soruda 'şiddet', 'dövülmek', 'intihar', 'taciz', 'tehdit', 'korku', 'yardım', 'tehlike', 'acil', 'kaçmak', 'kurtulmak' gibi acil müdahale gerektiren ifadeler varsa:
   -  Kısa ve güçlü bir empati ifadesiyle başla: "Yaşadığınız durumun ne kadar zor olduğunu anlıyorum ve güvende olmanız en önemli öncelik."
   -  Acil yardım hatlarını (183, 155, 112) öncelikli ve vurgulu olarak belirt: "HEMEN ŞU NUMARALARI ARAYABİLİRSİNİZ:"
   -  Kullanıcının belirttiği veya IP adresinden tespit edilebilen şehre göre en yakın ŞÖNİM, sığınma evi veya kadın danışma merkezi bilgilerini detaylı olarak paylaş (telefon, adres, çalışma saatleri)
   -  Yasal hakları ve koruma mekanizmalarını (6284 sayılı kanun, uzaklaştırma kararı) somut adımlarla açıkla: "Şu anda yapabileceğiniz adımlar:"
   -  Güvenlik planı oluşturmasına yardımcı ol: "Güvenliğiniz için şunları yapabilirsiniz: Önemli belgeleri hazır tutun, güvenli bir yere gitmek için plan yapın..."
   -  Psikolojik destek kaynaklarını belirt ve travma durumunda ilk yapılması gerekenleri açıkla

4. **Bilgi Sorularında Derinlik:** Bilgi sorularında sadece yüzeysel cevaplar verme. Örneğin "6284 sayılı kanun nedir?" sorusuna, kanunun tanımının yanı sıra, nasıl uygulandığı, hangi koruma tedbirlerini içerdiği ve başvuru süreçleri hakkında da bilgi ver.

5. **Alternatif Bakış Açıları:** Karmaşık durumlarda farklı seçenekler ve bakış açıları sun. Kullanıcıya karar verme sürecinde yardımcı ol, ancak onun yerine karar verme.

### BİLGİ SUNUMU ÇEŞİTLİLİĞİ ###
1. **Format Değişimi:** Bilgiyi her seferinde farklı formatlarda sun. Bazen paragraflar, bazen maddeler, bazen soru-cevap formatı, bazen adım adım yönergeler kullan.

2. **Vurgu Değişimi:** Her yanıtta farklı noktalara vurgu yap. Aynı bilgileri tekrar etmek yerine, her seferinde farklı yönleri öne çıkar.

3. **Detay Seviyesi:** Yanıtların detay seviyesini değiştir. Bazen özet bilgi, bazen derinlemesine analiz sun.

4. **Anlatım Tarzı:** Anlatım tarzını çeşitlendir. Bazen daha resmi, bazen daha samimi, bazen daha eğitici, bazen daha motive edici bir dil kullan.

### İLETİŞİM TARZI VE GÜÇLENDİRİCİ DİL ###

1. **Samimi ve Profesyonel Denge:** 
   - Resmi olmayan ama profesyonel bir dil kullan
   - Teknik terimlerden kaçın, ancak gerektiğinde açıklamalarıyla birlikte kullan
   - Hukuki terimleri basitleştirerek açıkla: "6284 sayılı kanun size şunları sağlıyor..."
   - Kullanıcının yaşına ve eğitim düzeyine uygun bir dil kullan

2. **Güçlendirici Dil ve Yaklaşım:** 
   - Kullanıcıyı asla pasif bir mağdur olarak gösterme
   - Her zaman haklarını arayan aktif bir birey olarak gör
   - "Yapabilirsiniz", "hakkınız var", "seçeneğiniz var", "karar sizin", "kontrol sizdedir" gibi güçlendirici ifadeler kullan
   - Suçluluk ve utancı azaltan ifadeler kullan: "Yaşadıklarınız sizin suçunuz değil", "Yardım istemek cesaret ister"
   - Küçük adımları kutla ve takdir et: "Yardım aramak için attığınız bu adım çok değerli"

3. **Yapılandırılmış ve Kolay Anlaşılır Yanıtlar:** 
   - Karmaşık bilgileri basit, anlaşılır adımlara böl
   - Önemli bilgileri vurgulamak için kalın yazı ve maddeler kullan
   - Acil durumlarda kısa ve net talimatlar ver: "HEMEN 155'i ARAYIN"
   - Uzun yanıtlarda bile okunabilirliği korumak için paragrafları kısa tut

4. **Empatik ve Destekleyici Ton:** 
   - "Sizi anladığımı bilmenizi isterim", "Bu zorlu süreçte yanınızdayım", "Cesaretiniz için sizi takdir ediyorum" gibi insani ifadeler kullan
   - Kullanıcının duygularını onaylayan ifadeler kullan: "Korkmanız çok doğal", "Endişelenmeniz anlaşılabilir"
   - Umut veren ama gerçekçi mesajlar ver: "Bu durumdan çıkış var ve size yardımcı olabilecek kurumlar mevcut"
   - Kullanıcının gücünü ve dayanıklılığını vurgula: "Bu adımı atmanız büyük bir cesaret gösteriyor"

5. **Kültürel Duyarlılık:**
   - Türkiye'nin farklı bölgelerindeki kültürel farklılıkları dikkate al
   - Dini veya kültürel hassasiyetlere saygı göster
   - Farklı sosyoekonomik durumlar için uygulanabilir çözümler sun

Unutma: Sen sadece bir bilgi kaynağı değil, aynı zamanda bir destek sesi, bir umut ışığısın. Her yanıtın, bir kadının hayatında olumlu bir fark yaratabilir. Standart cevaplar yerine, her duruma özel, düşünce dolu ve destekleyici yanıtlar vermeye özen göster. Aynı soruya her seferinde farklı bir bakış açısı ve bilgi sunarak kullanıcıya daha kapsamlı destek sağla."""
        
        return {"role": "system", "content": system_content}

    def _create_openai_response(self, query: str, context_text: str, 
                             temperature: Optional[float] = None,
                             use_history: bool = True) -> str:
        """OpenAI API kullanarak yanıt oluştur

        Args:
            query (str): Kullanıcı sorusu
            context_text (str): Bağlam metni
            temperature (Optional[float], optional): Sıcaklık parametresi
            use_history (bool, optional): Sohbet geçmişi kullanılsın mı

        Returns:
            str: Oluşturulan yanıt
        """
        try:
            if not self.openai_api_key or not self.llm:
                raise ValueError("OpenAI API anahtarı veya model bulunamadı")

            # Sistem mesajını hazırla
            system_message = self._get_system_message()
            
            # Sohbet geçmişini ekle
            messages = [system_message]
            if use_history and self.conversation_history:
                messages.extend(self.conversation_history[-5:])  # Son 5 mesaj
            
            # Şiddet/intihar düşüncesi kontrolü
            query_lower = query.lower()
            crisis_type = None
            
            # Şiddet veya intihar düşüncesi var mı kontrol et
            for intent_type, keywords in self.CRISIS_KEYWORDS.items():
                if any(keyword in query_lower for keyword in keywords):
                    crisis_type = intent_type
                    break

            # Kriz durumu varsa özel sistem mesajı ve bağlam ekle
            if crisis_type:
                crisis_resources = "\n\nDestekleyici Kaynaklar:\n"
                for category, resources in self.SUPPORT_RESOURCES.items():
                    crisis_resources += f"\n{category}:\n"
                    for resource in resources:
                        crisis_resources += f"- {resource}\n"

                context_text += crisis_resources
                
                if crisis_type == "violence":
                    messages.append({"role": "system", "content": (
                        "Kullanıcı şiddet/saldırganlık dürtüleri ifade ediyor. "
                        "Empatik ve destekleyici ol. Öfke ve saldırganlık duygularını normalize et. "
                        "Yardım aramanın değerli bir adım olduğunu vurgula. "
                        "Profesyonel destek almayı teşvik et ve kaynakları paylaş."
                    )})
                else:  # suicide
                    messages.append({"role": "system", "content": (
                        "Kullanıcı intihar düşünceleri ifade ediyor. "
                        "Duygularını ciddiye al ve anlayışla karşıla. "
                        "Yalnız olmadığını vurgula. "
                        "Acil yardım hatlarını öncelikle paylaş ve profesyonel destek almaya teşvik et."
                    )})

            # Kullanıcı sorusunu ve bağlamı ekle
            user_message = f"Bağlam:\n{context_text}\n\nSoru: {query}"
            messages.append({"role": "user", "content": user_message})
            
            # OpenAI'dan yanıt al
            response = self.llm.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=1024,
                top_p=0.9,
                frequency_penalty=0.6,
                presence_penalty=0.6
            )
            
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenAI yanıtı oluşturulurken hata: {str(e)}")
            return "Üzgünüm, yanıt oluşturulurken bir hata meydana geldi."

        
    def generate_response(self, query: str, context: Optional[List[Document]] = None,
            max_new_tokens: int = 1024, temperature: Optional[float] = None,
            top_p: float = 0.9, do_sample: bool = True,
            use_history: bool = True) -> Tuple[str, List[str]]:
        """Kullanıcı sorusuna yanıt oluştur
        
        Args:
            query (str): Kullanıcı sorusu
            context (Optional[List[Document]], optional): Bağlam belgeleri
            max_new_tokens (int, optional): Maksimum yanıt uzunluğu
            temperature (Optional[float], optional): Sıcaklık parametresi
            top_p (float, optional): Top-p örnekleme parametresi
            do_sample (bool, optional): Örnekleme yapılsın mı
            use_history (bool, optional): Sohbet geçmişi kullanılsın mı
            
        Returns:
            Tuple[str, List[str]]: Yanıt ve kullanılan kaynaklar
        """
        # Konu dışı soru kontrolü yap
        if not self._is_relevant_query(query):
            return "Üzgünüm, bu konu hakkında bilgim yok. Ben kadına yönelik şiddet, yasal haklar ve psikolojik destek konularında yardımcı olabilirim.", []
        try:
            # Sığınma evleri veya danışma merkezleri hakkında soru sorulup sorulmadığını kontrol et
            shelter_keywords = [
                "sığınma", "sığınak", "sığınmaevi", "konukevi", "kadın sığınmaevi", 
                "sığınabileceğim", "sığınabilir miyim", "nereye gidebilirim", "nerede kalabilirim",
                "şönim", "koruma", "barınma", "kadın konukevi", "gidebileceğim yer", "barınma kuruluşları"
            ]
            
            counseling_keywords = [
                "danışma", "destek", "merkez", "danışma merkezi", "kadın danışma", 
                "danışabileceğim", "yardım alabileceğim", "başvurabileceğim", "destek hattı",
                "danışmanlık", "mor çatı", "kadın dayanışma", "kadav"
            ]
            
            # Şehir isimlerini kontrol et
            city = self._find_city_in_query(query)
            
            # Şehir kontrolü
            city_mentioned = None
            if hasattr(self, 'siginma_evleri') and self.siginma_evleri:
                for shelter in self.siginma_evleri:
                    city = shelter.get("city", "").lower()
                    if city in query.lower():
                        city_mentioned = city
                        break
            
            # Sorgu türü kontrolü
            is_shelter_query = any(keyword in query.lower() for keyword in self.SHELTER_KEYWORDS)
            is_counseling_query = any(keyword in query.lower() for keyword in self.COUNSELING_KEYWORDS)
            
            # Bağlam hazırlama
            if context is None:
                context = self.get_context(query)
            
            context_text = ""
            sources = []
            
            # Sığınma evi veya danışma merkezi sorgusu işleme
            if (is_shelter_query or is_counseling_query) and hasattr(self, 'siginma_evleri'):
                shelter_info = ""
                counseling_info = ""
                
                # Şehre özel bilgiler
                if city_mentioned:
                    # Sığınma evi bilgileri
                    shelter = next(
                        (s for s in self.siginma_evleri 
                         if self._normalize_city(s.get("city", "")) == self._normalize_city(city_mentioned)),
                        None
                    )
                    
                    if shelter:
                        shelter_info = f"**{shelter.get('city')} ŞÖNİM:**\n"
                        shelter_info += self._format_contact_info(shelter)
                        sources.append("siginma_evleri.json")
                    
                    # Danışma merkezi bilgileri
                    if hasattr(self, 'danisma_merkezleri'):
                        city_centers = [
                            c for c in self.danisma_merkezleri
                            if self._normalize_city(c.get("city", "")) == self._normalize_city(city_mentioned)
                        ]
                        
                        if city_centers:
                            counseling_info = f"**{city_mentioned.title()} Şehrindeki Kadın Danışma Merkezleri:**\n\n"
                            for center in sorted(city_centers, 
                                               key=lambda x: x.get('type', '') != 'Bağımsız')[:3]:
                                counseling_info += f"**{center.get('name')}**\n"
                                counseling_info += f"Türü: {center.get('type', 'Belirtilmemiş')}\n"
                                counseling_info += self._format_contact_info(center)
                                counseling_info += f"Telefon: {', '.join(center.get('phone', []))}\n"
                                counseling_info += f"Adres: {center.get('address')}\n"
                                if center.get('website'):
                                    counseling_info += f"Web: {center.get('website')}\n"
                                counseling_info += "\n"
                            sources.append("kadin_danisma_merkezleri.json")
                else:
                    # Şehir belirtilmemişse önemli merkezleri göster
                    # Önemli şehirlerdeki sığınma evlerini ekle
                    big_cities = ["istanbul", "ankara", "izmir", "bursa", "antalya"]
                    shelter_info += "**Önemli Şehirlerdeki ŞÖNİM Bilgileri:**\n\n"
                    for city_name in big_cities:
                        for shelter in self.siginma_evleri:
                            if shelter.get("city", "").lower() == city_name:
                                shelter_info += f"**{shelter.get('city')} ŞÖNİM:**\n"
                                shelter_info += f"Telefon: {', '.join(shelter.get('phone', []))}\n"
                                shelter_info += f"Adres: {shelter.get('address')}\n\n"
                                break
                    sources.append("siginma_evleri.json")
                    
                    # Bağımsız kadın örgütlerini ekle
                    if hasattr(self, 'danisma_merkezleri') and self.danisma_merkezleri:
                        counseling_info += "**Bağımsız Kadın Örgütleri:**\n\n"
                        for center in self.danisma_merkezleri:
                            if center.get("type") == "Bağımsız":
                                counseling_info += f"**{center.get('name')}:**\n"
                                counseling_info += f"Telefon: {', '.join(center.get('phone', []))}\n"
                                counseling_info += f"Adres: {center.get('address')}\n"
                                if center.get('website'):
                                    counseling_info += f"Web: {center.get('website')}\n"
                                counseling_info += "\n"
                                if len(counseling_info.split('\n')) > 15:  # En fazla 3 merkez göster
                                    break
                        sources.append("kadin_danisma_merkezleri.json")
                
                # Acil yardım hatlarını ekle
                emergency_info = "\n**Acil Yardım Hatları:**\n"
                emergency_info += "- Alo 183 Sosyal Destek Hattı\n"
                emergency_info += "- 155 Polis İmdat\n"
                emergency_info += "- 156 Jandarma\n"
                emergency_info += "- 112 Acil Yardım\n"
                
                # JSON verilerini bağlam metnine ekle
                if shelter_info:
                    context_text += shelter_info + "\n\n"
                if counseling_info:
                    context_text += counseling_info + "\n\n"
                context_text += emergency_info + "\n\n"
                
                # Web kaynaklarını ekle
                context_text += "Web kaynaklarından elde edilen ek bilgiler:\n\n"
            
            # Web kaynaklarından bilgileri ekle
            for doc in context:
                context_text += doc.page_content + "\n\n"
                source = doc.metadata.get("source", "")
                
                # Kaynak bilgisini düzenle
                if source and source not in sources:
                    # PDF dosya yolunu işle
                    if ".pdf" in source.lower():
                        # Sadece dosya adını al
                        pdf_name = source.split('/')[-1].split('\\')[-1]
                        if pdf_name not in sources:
                            sources.append(pdf_name)
                    # Web URL'leri ve diğer kaynaklar
                    elif "http" in source.lower() or "www." in source.lower():
                        # Web URL'lerini doğrudan ekle
                        if source not in sources:
                            sources.append(source)
                    # Diğer kaynaklar
                    else:
                        sources.append(source)
                    
            # Sohbet geçmişi
            conversation_context = ""
            if use_history and self.conversation_history:
                conversation_context = self.get_conversation_context()
                
            # OpenAI ile yanıt oluştur
            try:
                # OpenAI API anahtarı yoksa uygun bir mesaj döndür
                if not self.openai_api_key or not self.llm:
                    self.logger.warning("OpenAI API anahtarı bulunamadığı için yanıt üretilemiyor.")
                    return (
                        "Üzgünüm, şu anda OpenAI API anahtarı bulunamadığı için sorularınızı yanıtlayamıyorum. "
                        "Lütfen .env dosyasına geçerli bir OpenAI API anahtarı ekleyin."
                    ), []
                
                # Sistem mesajını _get_system_message fonksiyonundan al
                system_message = self._get_system_message()
                
                user_message = f"Bağlam:\n{context_text}\n"
                
                # Sohbet geçmişini ekle
                if conversation_context:
                    user_message += f"\n{conversation_context}"
                    
                user_message += f"\n\nSoru: {query}"
                
                # OpenAI API ile yanıt oluştur
                messages = [
                    system_message,
                    {"role": "user", "content": user_message}
                ]
                
                # Benzer soru kontrolü yap ve çeşitlilik sağla
                try:
                    is_similar, similar_question = self._is_similar_question(query)
                except Exception as e:
                    self.logger.error(f"Benzer soru kontrolü yapılırken hata: {str(e)}")
                    is_similar, similar_question = False, ""
                
                # Sıcaklık değerini ayarla
                actual_temperature = temperature if temperature is not None else self.temperature
                
                # Benzer soru için çeşitlilik sağla
                if is_similar:
                    # Sıcaklık değerini artır (daha çeşitli yanıtlar için)
                    actual_temperature = min(actual_temperature + 0.2, 1.0)  # Max 1.0 olacak şekilde artır
                    # Top_p değerini artır (daha geniş kelime dağılımı için)
                    actual_top_p = 0.98
                    self.logger.info(f"Benzer soru için çeşitlilik parametreleri: temperature={actual_temperature}, top_p={actual_top_p}")
                else:
                    actual_top_p = top_p
                
                # Daha kapsamlı yanıtlar için parametreleri ayarla
                response = self.llm.invoke(
                    messages,
                    temperature=actual_temperature,  # Çeşitlilik için ayarlanmış sıcaklık
                    max_tokens=2048,  # Daha uzun yanıtlar için
                    top_p=actual_top_p,  # Çeşitlilik için ayarlanmış top_p
                )
                answer = response.content
                
                self.logger.info("OpenAI ile yanıt başarıyla oluşturuldu")
                
                # Eğer yanıt "bilgim yok" içeriyorsa boş kaynak listesi döndür
                if "Üzgünüm, bu konuda bilgim yok" in answer or "bilgim bulunmamaktadır" in answer.lower():
                    return answer, []
                    
                return answer, sources
            except Exception as e:
                self.logger.error(f"OpenAI ile yanıt oluşturulurken hata: {str(e)}")
                return f"Üzgünüm, bir hata oluştu: {str(e)}", []
                
        except Exception as e:
            self.logger.error(f"Yanıt oluşturulurken hata: {str(e)}")
            return "Üzgünüm, yanıt oluşturulurken bir hata meydana geldi.", []
            
    def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        """Sorgu için vektör veritabanından bağlam dökümanları alır"""
        if not self.vector_store:
            self.logger.warning("Vektör veritabanı yüklenmedi: Bağlam alınamıyor")
            return []
        try:
            self.logger.info(f"Sorgu için bağlam alınıyor: {query}")
            
            # Vektör veritabanı hakkında bilgi al
            if hasattr(self.vector_store, "_collection") and hasattr(self.vector_store._collection, "count"):
                count = self.vector_store._collection.count()
                self.logger.info(f"Vektör veritabanında {count} döküman bulunuyor")
                
                # Eğer döküman yoksa uyarı ver
                if count == 0:
                    self.logger.warning("Vektör veritabanında hiç döküman bulunmuyor. Lütfen prepare_data.py dosyasını çalıştırarak vektör veritabanını oluşturun.")
                    return []
            
            # Benzerlik araması yap
            results = self.vector_store.similarity_search(query, k=k)
            self.logger.info(f"{len(results)} bağlam dökümanı bulundu")
            
            # Bulunan dökümanları logla
            if results:
                for i, doc in enumerate(results):
                    source = doc.metadata.get("source", "Bilinmeyen kaynak")
                    self.logger.info(f"Döküman {i+1}: {source} - {len(doc.page_content)} karakter")
            else:
                self.logger.warning(f"Sorgu için hiç döküman bulunamadı: {query}")
                
            return results
        except Exception as e:
            self.logger.error(f"Bağlam alınırken hata: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
            
    def _load_shelter_data(self):
        """Sığınma evleri ve danışma merkezleri verilerini yükle"""
        try:
            # Proje dizinini al
            project_dir = Path(__file__).resolve().parent
            data_dir = project_dir / "data"
            
            # Sığınma evleri JSON dosyasını yükle
            siginma_evleri_path = data_dir / "siginma_evleri.json"
            if siginma_evleri_path.exists():
                with open(siginma_evleri_path, "r", encoding="utf-8") as f:
                    self.siginma_evleri = json.load(f)
                self.logger.info(f"{len(self.siginma_evleri)} adet sığınma evi bilgisi yüklendi")
            else:
                self.logger.warning(f"Sığınma evleri dosyası bulunamadı: {siginma_evleri_path}")
            
            # Kadın danışma merkezleri JSON dosyasını yükle
            danisma_merkezleri_path = data_dir / "kadin_danisma_merkezleri.json"
            if danisma_merkezleri_path.exists():
                with open(danisma_merkezleri_path, "r", encoding="utf-8") as f:
                    danisma_data = json.load(f)
                    
                    # JSON formatını kontrol et ve uygun şekilde işle
                    if isinstance(danisma_data, list):
                        # Dizi formatı - doğrudan kullan
                        self.danisma_merkezleri = danisma_data
                        self.logger.info(f"{len(self.danisma_merkezleri)} adet kadın danışma merkezi bilgisi yüklendi")
                    elif isinstance(danisma_data, dict):
                        # Şehir bazlı nesne formatı - düz listeye dönüştür
                        flat_list = []
                        for city, centers in danisma_data.items():
                            for center in centers:
                                # Şehir bilgisini ekle
                                if "city" not in center:
                                    center["city"] = city
                                flat_list.append(center)
                        self.danisma_merkezleri = flat_list
                        self.logger.info(f"{len(self.danisma_merkezleri)} adet kadın danışma merkezi bilgisi yüklendi (şehir bazlı format)")
                    else:
                        self.logger.warning("Kadın danışma merkezleri JSON formatı tanınmadı")
                        self.danisma_merkezleri = []
            else:
                self.logger.warning(f"Kadın danışma merkezleri dosyası bulunamadı: {danisma_merkezleri_path}")
                self.danisma_merkezleri = []
                
        except Exception as e:
            self.logger.error(f"Sığınma evleri ve danışma merkezleri verilerini yüklerken hata: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """İki sorgu arasındaki benzerliği hesaplar (0-1 arası)"""
        try:
            # Sorguları normalize et
            import re
            import string
            from difflib import SequenceMatcher
            
            normalized_query1 = self._normalize_text(query1)
            normalized_query2 = self._normalize_text(query2)
            
            # Basit benzerlik hesaplama (0-1 arası)
            similarity = SequenceMatcher(None, normalized_query1, normalized_query2).ratio()
            return similarity
        except Exception as e:
            self.logger.error(f"Benzerlik hesaplanırken hata: {str(e)}")
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Metni normalize eder (küçük harf ve noktalama işaretlerini kaldırma)"""
        import re
        import string
        # Türkçe karakterleri koru ama noktalama işaretlerini kaldır
        text = text.lower()
        text = re.sub(r'[' + string.punctuation + ']', '', text)
        return text
        
    def _is_similar_question(self, query: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """Verilen sorgunun daha önce sorulmuş sorulara benzer olup olmadığını kontrol eder"""
        if not self.question_history:
            return False, ""
        
        # En yüksek benzerliği ve ilgili soruyu bul
        max_similarity = 0.0
        most_similar_question = ""
        
        for prev_query in self.question_history:
            similarity = self._calculate_similarity(query, prev_query)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_question = prev_query
        
        # Benzerlik eşiğini aşıyorsa, benzer soru olarak kabul et
        if max_similarity >= threshold:
            self.logger.info(f"Benzer soru tespit edildi: {most_similar_question} (benzerlik: {max_similarity:.2f})")
            return True, most_similar_question
        
        return False, ""
        
    def _is_relevant_query(self, query: str) -> bool:
        """Sorgunun chatbot'un konusu ile ilgili olup olmadığını kontrol eder"""
        # Konu ile ilgili anahtar kelimeler
        relevant_keywords = [
            # Kadına yönelik şiddet ile ilgili kelimeler
            "şiddet", "taciz", "tecavüz", "istismar", "darp", "dayak", "tehdit", "takip", 
            "kadın", "kadına", "kadına şiddet", "aile içi şiddet", "ev içi şiddet", "fiziksel şiddet", 
            "psikolojik şiddet", "cinsel şiddet", "ekonomik şiddet", "dijital şiddet", "ısrarlı takip",
            # Yasal haklar ile ilgili kelimeler
            "hak", "yasal", "kanun", "hukuk", "dava", "şikayet", "boşanma", "nafaka", "velayet", 
            "uzaklaştırma", "koruma", "tedbir", "6284", "istanbul sözleşmesi", "cedaw", 
            # Psikolojik destek ile ilgili kelimeler
            "destek", "psikolojik", "terapi", "travma", "danışma", "yardım", "tedavi", "iyileşme",
            "tssb", "depresyon", "anksiyete", "panik", "korku", "güvenlik", "güvenli",
            # Sığınma ve destek merkezleri ile ilgili kelimeler
            "sığınma", "sığınak", "şönim", "merkez", "danışma merkezi", "mor çatı", "kadın dayanışma",
            "kadav", "acil", "telefon", "hat", "183", "155", "156", "112",
            # Şiddet uygulayan kişiler için eklenen kelimeler
            "öfke", "kontrol", "saldırganlık", "dövüyorum", "vuruyorum", "zarar veriyorum", 
            "şiddet uyguluyorum", "kendimi tutamıyorum", "öfke kontrolü", "agresif", "sinirliyim",
            "pişmanlık", "kendimi kontrol edemiyorum", "kendime zarar", "intihar", "zarar vermek",
            "tedavi olmak", "yardım almak", "terapist", "psikiyatrist", "psikolog"
        ]
        
        # Sorgu metni içinde ilgili anahtar kelimelerden herhangi biri var mı kontrol et
        normalized_query = self._normalize_text(query.lower())
        for keyword in relevant_keywords:
            if keyword.lower() in normalized_query:
                return True
                
        # Şiddet uygulama veya kendine zarar verme içeren sorgular için özel kontrol
        violence_patterns = [
            "dövüyorum", "vuruyorum", "şiddet uyguluyorum", "zarar veriyorum", "kendimi tutamıyorum",
            "kendime zarar", "intihar", "öldürmek", "zarar vermek", "kendimi kontrol edemiyorum"
        ]
        
        for pattern in violence_patterns:
            if pattern in normalized_query:
                return True
                
        # Vektör veritabanından ilgili dokümanlar var mı kontrol et
        try:
            if self.vector_store:
                docs = self.vector_store.similarity_search(query, k=3)
                if docs and len(docs) > 0:
                    # Benzerlik skoru yeterince yüksek mi kontrol et
                    similarity_scores = [doc.metadata.get("score", 0) for doc in docs if "score" in doc.metadata]
                    if similarity_scores and max(similarity_scores, default=0) > 0.7:
                        return True
        except Exception as e:
            self.logger.error(f"Vektör veritabanı sorgusu sırasında hata: {str(e)}")
            
        # Hiçbir ilgili anahtar kelime veya benzer doküman bulunamadıysa konu dışı kabul et
        return False
    
    def reset_chat(self):
        """Sohbet geçmişini sıfırla"""
        self.conversation_history = []
        self.question_history = []
        self.used_sources = {}
        self.logger.info("Sohbet geçmişini sıfırladım")
