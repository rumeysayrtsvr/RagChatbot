#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kadına Yönelik Şiddet ve Destek Chatbot API
FastAPI kullanarak chatbot'u bir REST API olarak sunar
"""

import os
import logging
import uvicorn
from typing import List, Dict, Optional
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# RAG Chatbot'u import et
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

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Kadına Yönelik Şiddet ve Destek Chatbot API",
    description="Kadına yönelik şiddet, yasal haklar ve psikolojik destek konularında bilgi sağlayan bir chatbot API'si",
    version="1.0.0"
)

# CORS ayarları - tüm kaynaklardan gelen isteklere izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (geliştirme için)
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver
    allow_headers=["*"],  # Tüm başlıklara izin ver
)

# Kullanıcı sohbet geçmişlerini saklamak için sözlük
chat_histories = {}

# Chatbot'u başlat
chatbot = None

# API için veri modelleri
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class ResetRequest(BaseModel):
    user_id: str = "default_user"

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """API başlatıldığında çalışacak fonksiyon"""
    global chatbot
    try:
        # Vektör veritabanı dizinini kontrol et
        if not VECTOR_STORE_DIR.exists():
            logger.warning(f"Vektör veritabanı dizini bulunamadı: {VECTOR_STORE_DIR}")
            raise HTTPException(status_code=500, detail="Vektör veritabanı bulunamadı")
        
        # Chatbot'u başlat
        chatbot = RAGChatbot(
            vector_store_path=str(VECTOR_STORE_DIR),
            collection_name=collection_name
        )
        
        # OpenAI API anahtarını kontrol et
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY bulunamadı!")
            raise HTTPException(status_code=500, detail="OpenAI API anahtarı bulunamadı")
        
        logger.info("Chatbot başarıyla başlatıldı")
    except Exception as e:
        logger.error(f"Chatbot başlatılırken hata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chatbot başlatılamadı: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Kullanıcı sorusunu yanıtla"""
    global chatbot
    
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot başlatılamadı")
    
    try:
        # Kullanıcı geçmişini kontrol et
        user_id = request.user_id
        if user_id not in chat_histories:
            chat_histories[user_id] = []
        
        # Chatbot'a soruyu sor
        response, sources = chatbot.chat(request.query)
        
        # Sohbet geçmişini güncelle
        chat_histories[user_id].append({"role": "user", "content": request.query})
        chat_histories[user_id].append({"role": "assistant", "content": response})
        
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        logger.error(f"Soru yanıtlanırken hata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Soru yanıtlanırken hata: {str(e)}")

@app.post("/reset", response_model=StatusResponse)
async def reset_chat(request: ResetRequest):
    """Sohbet geçmişini sıfırla"""
    global chatbot
    
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot başlatılamadı")
    
    try:
        user_id = request.user_id
        if user_id in chat_histories:
            chat_histories[user_id] = []
        
        chatbot.reset_chat()
        return StatusResponse(status="success", message="Sohbet geçmişi sıfırlandı")
    except Exception as e:
        logger.error(f"Sohbet sıfırlanırken hata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sohbet sıfırlanırken hata: {str(e)}")

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """API sağlık kontrolü"""
    global chatbot
    
    if not chatbot:
        return StatusResponse(status="error", message="Chatbot başlatılamadı")
    
    return StatusResponse(status="success", message="API çalışıyor")

if __name__ == "__main__":
    # API'yi çalıştır
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
