# Panduan Menjalankan FastAPI Backend (AI System)

Dokumen ini berisi panduan untuk menjalankan dan melakukan pengujian pada server FastAPI khusus untuk project **AI System (RKS RAG Agent)**.

---

## 📋 Prasyarat

1. Pastikan Anda sudah membuka terminal dan berada di dalam direktori `AI_System`.
2. Pastikan file `config.py` sudah memiliki `REMOTE_LLM_URL` (URL ngrok) yang aktif.
3. Pastikan dependensi sudah terinstall (jika belum, jalankan `uv sync`).

---

## 🚀 Cara Menjalankan Server API

Anda dapat menjalankan server FastAPI menggunakan salah satu dari dua metode berikut melalui terminal:

### Metode 1: Menggunakan Script Python (Bawaan)

File `api.py` sudah dilengkapi dengan _entry point_ uvicorn bawaan. Cukup jalankan:

```bash
uv run python api.py
```

_Catatan: Server akan berjalan di port `8000` tanpa fitur hot-reload._

### Metode 2: Menggunakan Uvicorn (Hot-Reload)

Sangat direkomendasikan jika Anda sedang dalam tahap _development_ atau sering mengubah kode. Server akan otomatis me-restart jika ada file yang di-save:

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 Mengakses Antarmuka API (Swagger UI)

FastAPI memiliki fitur _auto-generated documentation_ yang sangat memudahkan Anda untuk menguji (test) API secara langsung melalui browser.

Setelah server menyala (muncul tulisan _Application startup complete_ di terminal), buka browser Anda dan akses:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

Di halaman ini, Anda bisa mencoba endpoint API tanpa perlu aplikasi eksternal seperti Postman.

---

## 📝 Penggunaan Endpoint Utama

### `POST /api/generate-rks`

Endpoint ini memproses ekstraksi file BOQ/PDF, menyusun struktur dokumen, dan men-generate RKS (`.docx`) menggunakan agen AI.

**Cara Test via Swagger UI (`/docs`):**

1. Buka [http://localhost:8000/docs](http://localhost:8000/docs).
2. Klik _dropdown_ pada endpoint `POST /api/generate-rks`.
3. Klik tombol **"Try it out"** di pojok kanan atas.
4. Isi parameter form-data berikut:
   - **`jenis_pekerjaan`**: (Contoh: `"Pengecatan Pipa"`)
   - **`detail_pekerjaan`**: (Contoh: `"Pengecatan pipa distribusi sepanjang 2KM"`)
   - **`lokasi`**: (Contoh: `"Terminal BBM"`)
   - **`file`**: Klik tombol **"Choose File"** dan pilih file PDF BOQ (contoh dari data dummy Anda).
5. Klik tombol **"Execute"**.
6. **Hasil:** Tunggu beberapa saat hingga AI selesai bekerja. API akan langsung men-download / mengembalikan file Word `.docx` hasil generate.

---

## ⚠️ Troubleshooting Umum

- **Port 8000 already in use:**
  Jika port bentrok dengan aplikasi lain, gunakan port lain:
  `uv run uvicorn api:app --reload --port 8080`
- **Internal Server Error (500) saat _Execute_:**
  Pastikan tunnel ngrok Anda di Kaggle/Colab tidak kedaluwarsa. Jika LLM jarak jauh terputus atau _timeout_, proses RAG (Retrieval) dan pembuatan _draft_ di `tools.py` akan gagal.
