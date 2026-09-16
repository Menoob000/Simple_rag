import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from RAG.summarizer import extract_text_from_pdfs, summarize_context
from RAG.tools import generate_rks_document, export_rks_to_docx

app = FastAPI(title="Pertamina RKS AI System - API")

@app.post("/api/generate-rks")
async def generate_rks(
    jenis_pekerjaan: str = Form(...),
    detail_pekerjaan: str = Form(...),
    lokasi: str = Form(""),
    resiko_csms: str = Form("MEDIUM"),
    nomor_dokumen: str = Form("RKS-[KODE]-[TAHUN]"),
    nama_perusahaan: str = Form("PT PERTAMINA"),
    file: UploadFile = File(..., description="Upload BOQ/Context PDF file here")
):
    try:
        context_summary = ""
        
        # 1. Process files and summarize context
        pdf_bytes_list = []
        if file and file.filename and file.filename.lower().endswith('.pdf'):
            print(f"Receiving file {file.filename} for context...")
            content = await file.read()
            pdf_bytes_list.append(content)
            
        if pdf_bytes_list:
                print("Extracting text from PDFs...")
                raw_text = extract_text_from_pdfs(pdf_bytes_list)
                print("Summarizing context using LLM...")
                context_summary = summarize_context(raw_text)
                print(f"Context Summary:\n{context_summary}\n")

        # 2. Generate RKS Draft
        print("Generating RKS document...")
        gen_result = generate_rks_document.invoke({
            "jenis_pekerjaan": jenis_pekerjaan,
            "detail_pekerjaan": detail_pekerjaan,
            "lokasi": lokasi,
            "context_summary": context_summary,
            "resiko_csms": resiko_csms,
            "nomor_dokumen": nomor_dokumen,
            "nama_perusahaan": nama_perusahaan
        })

        if "Error" in gen_result or "Tidak ditemukan" in gen_result:
            raise HTTPException(status_code=400, detail=gen_result)

        # 3. Export to Docx
        print("Exporting to DOCX...")
        export_result = export_rks_to_docx.invoke({"output_filename": f"RKS_{jenis_pekerjaan.replace(' ', '_')}.docx"})
        
        if "Error" in export_result or "Tidak ada draft" in export_result:
            raise HTTPException(status_code=400, detail=export_result)
            
        # Parse the output path from export_result
        lines = export_result.split('\n')
        file_path = ""
        for line in lines:
            if "📄 File:" in line:
                file_path = line.split("📄 File:")[1].strip()
                break

        if not file_path or not os.path.exists(file_path):
             return {
                 "status": "success", 
                 "message": "Generated but could not find docx path", 
                 "raw_generation_result": gen_result,
                 "raw_export_result": export_result
             }

        # Return FileResponse to download the docx immediately
        return FileResponse(
            path=file_path, 
            filename=os.path.basename(file_path), 
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
