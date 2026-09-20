# Thai RAG API

API ต้นแบบสำหรับถามตอบจากฐานข้อมูลความรู้ภาษาไทย โดยใช้ Ollama บนเครื่อง

- โมเดลตอบคำถาม: `qwen3.5:9b-q4_K_M`
- โมเดลค้นเอกสาร: `bge-m3`
- API framework: FastAPI
- ที่เก็บข้อมูลต้นแบบ: SQLite ใน `data/rag.sqlite3`

## 1. เตรียม Ollama

ต้องมีโมเดลสองตัวนี้ก่อน:

```powershell
ollama pull qwen3.5:9b-q4_K_M
ollama pull bge-m3
```

เช็กว่า Ollama ทำงาน:

```powershell
ollama list
```

## 2. ติดตั้ง Python packages

แนะนำให้สร้าง virtual environment ก่อน:

```powershell
cd "C:\Users\Lenovo\Desktop\My Project\thai-rag-api"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. ตั้งค่า API key

สำหรับลองบนเครื่องจะยังไม่ตั้งก็ได้ แต่ถ้าจะให้คนอื่นใช้ต้องตั้ง:

```powershell
$env:RAG_API_KEY = "my-secret-key"
```

## 4. เปิด API

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

เปิดเอกสาร API ได้ที่:

```text
http://127.0.0.1:8000/docs
```

เปิดหน้าเว็บจำลองสำหรับผู้ใช้ได้ที่:

```text
http://127.0.0.1:8000/
```

## 5. ใส่ข้อมูลเข้าฐานความรู้

ในระบบจริง ข้อมูลส่วนนี้จะมาจาก database ของเรา เช่นตารางประกาศ/ข่าว/FAQ
สำหรับต้นแบบนี้สามารถเพิ่มข้อความผ่าน API ได้:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/knowledge/text" `
  -H "Content-Type: application/json" `
  -H "x-api-key: my-secret-key" `
  -d "{\"title\":\"ประกาศตัวอย่าง\",\"source_ref\":\"db:announcements:1\",\"text\":\"ข้อความความรู้ที่ต้องการให้ AI ใช้อ้างอิง\"}"
```

ถ้ายังไม่ได้ตั้ง `RAG_API_KEY` ให้เอาบรรทัด `-H "x-api-key: ..."` ออก

## 6. เครื่องมือช่วยสำหรับ admin: อัปโหลด PDF

ใช้หน้า `/docs` ง่ายสุด หรือใช้ PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -H "x-api-key: my-secret-key" `
  -F "file=@C:\Users\Lenovo\Downloads\1. ประกาศหลักเกณฑ์-2569.pdf"
```

ถ้ายังไม่ได้ตั้ง `RAG_API_KEY` ให้เอาบรรทัด `-H "x-api-key: ..."` ออก

หรือโหลด PDF เข้า database ล่วงหน้าผ่านสคริปต์ admin:

```powershell
python scripts/load_pdf.py "C:\Users\Lenovo\Downloads\1. ประกาศหลักเกณฑ์-2569.pdf"
```

## 7. ถามคำถาม

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -H "x-api-key: my-secret-key" `
  -d "{\"question\":\"ประกาศนี้เกี่ยวกับเรื่องอะไร สรุปสั้น ๆ\"}"
```

## หมายเหตุเรื่อง PDF สแกน

ถ้า API ตอบว่า `No selectable text found` แปลว่า PDF เป็นไฟล์สแกนหรือรูปภาพ ต้อง OCR ก่อน
แล้วค่อยอัปโหลดไฟล์ที่มีข้อความจริงเข้าระบบ

## เส้นทางต่อยอด

เมื่อต้องเปิดให้คนอื่นใช้จริง ควรเพิ่มอย่างน้อย:

- ฐานข้อมูลจริง เช่น PostgreSQL + pgvector หรือ Qdrant
- API key แยกตามลูกค้า
- rate limit
- logging และ usage tracking
- queue สำหรับ ingest เอกสารใหญ่
- deploy บน Cloud VM ที่มี RAM/GPU พอ ไม่ควรเปิดจากโน้ตบุ๊กส่วนตัว
