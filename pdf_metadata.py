import os
import json
from pathlib import Path

def create_pdf_metadata(pdf_dir="data/pdfs", output_file="data/pdf_metadata.json"):
    """
    PDF dosyalarının metadata bilgilerini içeren bir JSON dosyası oluşturur.
    Bu dosya, PDF adı ve dosya yolu eşleştirmelerini içerir.
    """
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        print(f"PDF dizini bulunamadı: {pdf_dir}")
        return False
    
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    
    pdf_metadata = {}
    for pdf_file in pdf_files:
        # PDF dosyasının adı
        pdf_name = pdf_file.name
        # PDF dosyasının tam yolu
        pdf_path = str(pdf_file.absolute())
        
        pdf_metadata[pdf_name] = {
            "path": pdf_path,
            "name": pdf_name
        }
    
    # JSON dosyasını oluştur
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pdf_metadata, f, ensure_ascii=False, indent=4)
    
    print(f"PDF metadata JSON dosyası oluşturuldu: {output_file}")
    print(f"Toplam {len(pdf_metadata)} PDF dosyası kaydedildi")
    
    return True

if __name__ == "__main__":
    create_pdf_metadata()
