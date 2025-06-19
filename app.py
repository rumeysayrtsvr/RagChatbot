import os
import json
import logging
import webbrowser
import subprocess
import base64
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# LangChain importları
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local importlar
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_chatbot import RAGChatbot

# .env dosyasından API anahtarını yükle
load_dotenv()

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Proje dizinleri
PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Koleksiyon adı
collection_name = "women_rights"

# Log mesajını göster
logger.info(f"Proje dizini: {PROJECT_ROOT}")
logger.info(f"Vektör veritabanı dizini: {VECTOR_STORE_DIR}")
logger.info(f"Kullanılacak koleksiyon adı: {collection_name}")

# Vektör veritabanı dizinini kontrol et
if not VECTOR_STORE_DIR.exists():
    logger.warning(f"Vektör veritabanı dizini bulunamadı: {VECTOR_STORE_DIR}")
else:
    # Alt dizinleri listele
    subdirs = [d for d in VECTOR_STORE_DIR.iterdir() if d.is_dir()]
    logger.info(f"Vektör veritabanı koleksiyonları: {[d.name for d in subdirs]}")
    
    # women_rights koleksiyonu var mı kontrol et
    women_rights_dir = VECTOR_STORE_DIR / "women_rights"
    if women_rights_dir.exists():
        logger.info(f"women_rights koleksiyonu bulundu: {women_rights_dir}")
    else:
        logger.warning(f"women_rights koleksiyonu bulunamadı: {women_rights_dir}")

# Dil modeli için şablon
TURKISH_QA_PROMPT = """
Sen kadına yönelik şiddet, yasal haklar ve psikolojik destek konularında uzmanlaşmış bir Türkçe chatbot'sun.
Sana sorulan soruları aşağıdaki bağlam bilgilerini kullanarak yanıtla.
Eğer cevabı bilmiyorsan, "Bu konuda bilgim yok" de ve uydurma.
Yanıtını Türkçe olarak ver.

Bağlam:
{context}

Soru: {question}
Yanıt:"""

# Sohbet geçmişi için şablon
TURKISH_CHAT_PROMPT = """
Sen kadına yönelik şiddet, yasal haklar ve psikolojik destek konularında uzmanlaşmış bir Türkçe chatbot'sun.
Sana sorulan soruları aşağıdaki bağlam bilgilerini kullanarak yanıtla.
Eğer cevabı bilmiyorsan, "Bu konuda bilgim yok" de ve uydurma.
Yanıtını Türkçe olarak ver.

Sohbet Geçmişi:
{chat_history}

Bağlam:
{context}

Soru: {question}
Yanıt:"""

# Ensure directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

class ChatbotApp:
    def __init__(self, model_name="gpt-4o"):
        # .env dosyasından API anahtarını yükle
        load_dotenv()
        
        self.model_name = model_name
        self.chatbot = None
        self.chat_history = []
        
        # Proje dizinlerini tanımla
        self.project_root = Path(__file__).resolve().parent
        self.vector_store_dir = self.project_root / "vector_store"
        
        # Logging ayarları
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def initialize(self):
        """Chatbot'u başlat"""
        try:
            # OpenAI API anahtarını kontrol et
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            # Mevcut koleksiyonları kontrol et
            subdirs = [d for d in self.vector_store_dir.iterdir() if d.is_dir()]
            collection_names = [d.name for d in subdirs]
            
            # Mevcut koleksiyonu kullan - women_rights varsa onu, yoksa ilk koleksiyonu kullan
            if not collection_names:
                raise ValueError("Vektör veritabanında hiç koleksiyon bulunamadı!")
                
            actual_collection_name = "women_rights" if "women_rights" in collection_names else collection_names[0]
            
            self.logger.info(f"Kullanılacak koleksiyon: {actual_collection_name}")
            
            # RAG Chatbot'u başlat
            self.logger.info(f"RAG Chatbot başlatılıyor...")
            if openai_api_key:
                # OpenAI API ile RAG Chatbot'u başlat
                embedding_model_name = "text-embedding-3-small"  # OpenAI embedding modeli
                self.logger.info(f"OpenAI embedding modeli kullanılıyor: {embedding_model_name}")
            else:
                # HuggingFace embedding modeli kullan
                embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # HuggingFace embedding modeli
                self.logger.info(f"HuggingFace embedding modeli kullanılıyor: {embedding_model_name}")
                
            self.chatbot = RAGChatbot(
                vector_store_path=str(self.vector_store_dir),
                embedding_model=embedding_model_name,
                openai_model=self.model_name,
                vector_db_type="chroma",
                collection_name=actual_collection_name
            )
                
            self.logger.info("RAG Chatbot başarıyla başlatıldı")
        except Exception as e:
            self.logger.error(f"Başlatma sırasında hata: {str(e)}")
            raise
            
    def ask(self, query):
        """Kullanıcı sorusunu yanıtla"""
        if not query.strip():
            return "Lütfen bir soru sorun.", []
            
        try:
            # RAG Chatbot ile yanıt al
            if self.chatbot:
                # Yanıtı ve kaynakları al
                answer, sources = self.chatbot.chat(query)
                return answer, sources
            else:
                return "Chatbot başlatılamadı. Lütfen .env dosyasına OpenAI API anahtarınızı ekleyin.", []
        except Exception as e:
            self.logger.error(f"Soru yanıtlanırken hata: {str(e)}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}", []
    
    def reset_chat(self):
        """Sohbet geçmişini sıfırla"""
        if self.chatbot:
            self.chatbot.conversation_history = []
        self.chat_history = []

def initialize_chatbot():
    """Chatbot'u başlat"""
    try:
        # Vektör veritabanı koleksiyonlarını kontrol et
        if not VECTOR_STORE_DIR.exists():
            st.error(f"Vektör veritabanı dizini bulunamadı: {VECTOR_STORE_DIR}")
            logger.error(f"Vektör veritabanı dizini bulunamadı: {VECTOR_STORE_DIR}")
            return None
            
        # Mevcut koleksiyonları listele
        subdirs = [d for d in VECTOR_STORE_DIR.iterdir() if d.is_dir()]
        collection_names = [d.name for d in subdirs]
        
        if not collection_names:
            logger.error(f"Vektör veritabanında hiç koleksiyon bulunamadı!")
            st.error(f"Vektör veritabanında hiç koleksiyon bulunamadı!")
            st.info("Lütfen önce 'prepare_data.py' dosyasını çalıştırarak vektör veritabanını oluşturun.")
            return None
        
        # Mevcut koleksiyonu kullan - women_rights varsa onu, yoksa ilk koleksiyonu kullan
        actual_collection_name = collection_name if collection_name in collection_names else collection_names[0]
        logger.info(f"Kullanılacak koleksiyon: {actual_collection_name}")
        
        # Chatbot'u başlat
        chatbot = ChatbotApp()
        logger.info(f"RAG Chatbot {actual_collection_name} koleksiyonu ile başlatılıyor...")
        
        try:
            # Önemli: initialize metodunda collection_name parametresini güncellemeliyiz
            chatbot.initialize()
            logger.info("RAG Chatbot başarıyla başlatıldı")
            return chatbot
        except Exception as e:
            logger.error(f"RAG Chatbot başlatılamadı: {str(e)}")
            st.error(f"RAG Chatbot başlatılamadı: {str(e)}")
            return None
        
        logger.info("RAG Chatbot başarıyla başlatıldı")
        return chatbot
    except Exception as e:
        logger.error(f"RAG Chatbot başlatılamadı: {str(e)}")
        st.error(f"RAG Chatbot başlatılamadı: {str(e)}")
        st.error(f"Chatbot başlatılırken hata: {str(e)}")
        return None

def main():
    """Ana uygulama"""
    # Session state'i başlat
    if "sources_data" not in st.session_state:
        st.session_state.sources_data = {
            "pdf_sources": [],
            "web_sources": [],
            "other_sources": []
        }
    
    # Sayfa başlığı
    st.title("Kadına Yönelik Şiddet ve Destek Chatbot")
    
    # Yan panel - Sadece sohbeti sıfırlama butonu
    with st.sidebar:
        # Sohbeti sıfırlama butonu
        if st.button("Sohbeti Sıfırla"):
            if "chatbot" in st.session_state and st.session_state.chatbot:
                # Chatbot'un belleğini sıfırla
                st.session_state.chatbot.reset_chat()
                st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Size nasıl yardımcı olabilirim?"}]
                # Kaynakları da sıfırla
                st.session_state.sources_data = {
                    "pdf_sources": [],
                    "web_sources": [],
                    "other_sources": []
                }
                st.rerun()
    
    # Chatbot'u başlat
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    
    # Mesajları sakla
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Size nasıl yardımcı olabilirim?"}]
    
    # Yan panel - Sadece sohbeti sıfırlama butonu için kullanılıyor
    
    # Mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Kullanıcı girişi
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        # Chatbot'un başlatıldığını kontrol et
        if not st.session_state.chatbot:
            st.error("RAG Chatbot başlatılamadı. Lütfen vektör veritabanını ve API anahtarını kontrol edin.")
            st.stop()
        
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Chatbot yanıtını göster
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Yanıtı al
            response, sources = st.session_state.chatbot.ask(prompt)
            
            # Yanıtı göster
            message_placeholder.markdown(response)
            
            # PDF metadata'yı yükle
            try:
                # Proje dizinini al
                project_dir = Path(__file__).resolve().parent
                data_dir = project_dir / "data"
                pdf_metadata_path = data_dir / "pdf_metadata.json"
                
                with open(pdf_metadata_path, "r", encoding="utf-8") as f:
                    pdf_metadata = json.load(f)
            except Exception as e:
                pdf_metadata = {}
                print(f"PDF metadata yüklenemedi: {str(e)}")
            
            # Kaynakları göster
            if sources:
                # Kaynakları türlerine göre ayır
                pdf_sources = []
                web_sources = []
                other_sources = []
                
                for source in sources:
                    if ".pdf" in source.lower():
                        # PDF dosya yolundan sadece dosya adını al
                        pdf_name = source.split('/')[-1].split('\\')[-1]
                        if pdf_name not in pdf_sources and pdf_name in pdf_metadata:
                            pdf_sources.append(pdf_name)
                    elif "http" in source.lower() or "www." in source.lower():
                        if source not in web_sources:
                            web_sources.append(source)
                    else:
                        if source not in other_sources:
                            other_sources.append(source)
                
                # Kaynakları göster
                with st.expander("Kaynaklar", expanded=True):
                    # Tüm kaynakları tek bir başlık altında göster
                    all_sources = []
                    
                    # Web kaynaklarını ekle (JSON dosyaları hariç)
                    for url in web_sources:
                        # JSON dosyalarını gösterme
                        if not url.lower().endswith('.json') and not '/json' in url.lower():
                            all_sources.append({
                                "type": "web",
                                "content": url,
                                "display": f"- [{url}]({url})"
                            })
                    
                    # Diğer kaynakları ekle (JSON dosyaları hariç)
                    for source in other_sources:
                        # JSON dosyalarını gösterme
                        if not source.lower().endswith('.json') and not 'json' in source.lower():
                            all_sources.append({
                                "type": "other",
                                "content": source,
                                "display": f"- {source}"
                            })
                    
                    # PDF kaynaklarını ekle
                    for pdf_name in pdf_sources:
                        if pdf_name in pdf_metadata:
                            pdf_path = pdf_metadata[pdf_name]["path"]
                            try:
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                
                                # HTML ile PDF'i açan bir link oluştur
                                html_link = f'''
                                <div style="margin-bottom: 10px;">
                                    <a href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" 
                                       download="{pdf_name}" 
                                       target="_blank" 
                                       style="display: inline-block; padding: 0.25rem 0.75rem; 
                                              background-color: #f0f2f6; color: #262730; 
                                              border-radius: 0.25rem; text-decoration: none; 
                                              font-weight: 400; border: 1px solid rgba(49, 51, 63, 0.2);">
                                        {pdf_name}
                                    </a>
                                </div>
                                '''
                                
                                all_sources.append({
                                    "type": "pdf",
                                    "content": pdf_name,
                                    "display": html_link,
                                    "is_html": True
                                })
                            except Exception as e:
                                all_sources.append({
                                    "type": "pdf",
                                    "content": pdf_name,
                                    "display": f"- {pdf_name}",
                                    "is_html": False
                                })
                        else:
                            all_sources.append({
                                "type": "pdf",
                                "content": pdf_name,
                                "display": f"- {pdf_name}",
                                "is_html": False
                            })
                    
                    # Kaynakları göster
                    if all_sources:
                        for source in all_sources:
                            if source.get("is_html", False):
                                st.markdown(source["display"], unsafe_allow_html=True)
                            else:
                                st.markdown(source["display"])

        
        # Asistan mesajını ekle
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Kaynakları ayrıca sakla (görüntülemek için değil)
        if sources:
            if "sources" not in st.session_state:
                st.session_state.sources = {}
            st.session_state.sources[len(st.session_state.messages)-1] = sources

if __name__ == "__main__":
    main()
