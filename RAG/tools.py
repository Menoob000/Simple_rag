from langchain_core.tools import tool
from RAG.store import get_vector_store
from RAG.rks_draft import RKSDraft
from RAG.rks_template import extract_rks_structure, fill_rks_content, revise_section
from RAG.docx_exporter import render_to_docx
import config
from langchain_openai import ChatOpenAI
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize retriever once for tool usage
_retriever = get_vector_store().as_retriever(
    search_kwargs={"k": config.TOP_K_RESULTS}
)

# Global draft state (persists across tool calls within a session)
_active_draft = RKSDraft()


def _get_llm():
    """Get a fresh LLM instance for internal tool calls."""
    return ChatOpenAI(
        base_url=config.REMOTE_LLM_URL,
        api_key="ollama",
        model=config.LLM_MODEL_NAME,
        temperature=0,
        timeout=600,
        streaming=True,
        default_headers={"ngrok-skip-browser-warning": "true"},
    )


def _search_multiple(queries: list[str]) -> str:
    """Search knowledge base with multiple queries for broader coverage."""
    all_results = []
    seen_content = set()

    for query in queries:
        results = _retriever.invoke(query)
        for doc in results:
            # Deduplicate by content
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_results.append(doc)

    return "\n\n---\n\n".join(
        [f"[Context Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(all_results)]
    )


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal documentation and knowledge base for domain context."""
    results = _retriever.invoke(query)
    if not results:
        return "No relevant internal records found for this query."
    
    return "\n\n---\n\n".join(
        [f"[Context Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(results)]
    )


@tool
def generate_rks_document(
    jenis_pekerjaan: str,
    detail_pekerjaan: str,
    lokasi: str = "",
    context_summary: str = "",
    resiko_csms: str = "MEDIUM",
    nomor_dokumen: str = "RKS-[KODE]-[TAHUN]",
    nama_perusahaan: str = "PT PERTAMINA"
) -> str:
    """Generate draft RKS (Rencana Kerja dan Syarat-Syarat) secara utuh berdasarkan knowledge base.
    Struktur dan konten diekstrak secara dinamis dari dokumen RKS sejenis.
    AI secara otomatis menentukan judul pekerjaan, level risiko CSMS, 
    nama unit perusahaan, dan struktur bab yang sesuai jenis pekerjaan.
    Gunakan tool ini ketika user meminta untuk membuat/generate RKS baru.
    """
    global _active_draft

    try:
        # Step 1: Search knowledge base with multiple queries
        print("  📚 Tahap 1/3: Mencari dokumen referensi di knowledge base...")
        queries = [
            f"RKS {jenis_pekerjaan} struktur bab",
            f"rencana kerja syarat {jenis_pekerjaan} spesifikasi",
            f"RKS {jenis_pekerjaan} {lokasi}" if lokasi else f"RKS {jenis_pekerjaan} persyaratan",
        ]
        chunks = _search_multiple(queries)

        if not chunks or chunks == "":
            return "Tidak ditemukan dokumen RKS relevan di knowledge base. Pastikan ada dokumen PDF RKS di folder RAG/Data/."

        # Step 2: Extract structure using LLM (fast, ~15-20 seconds)
        print("  🏗️ Tahap 2/3: Mengekstrak struktur dokumen RKS...")
        llm = _get_llm()
        structure = extract_rks_structure(
            chunks=chunks,
            llm=llm,
            jenis_pekerjaan=jenis_pekerjaan,
            detail_pekerjaan=detail_pekerjaan,
            lokasi=lokasi,
            context_summary=context_summary,
            resiko_csms=resiko_csms,
            nomor_dokumen=nomor_dokumen,
            nama_perusahaan=nama_perusahaan
        )

        num_bab = len(structure.get("bab", []))
        print(f"  ✅ Struktur berhasil diekstrak: {num_bab} bab ditemukan")

        # Step 3: Fill content PER CHAPTER (each ~40-80 seconds)
        def on_progress(current, total, title):
            print(f"  📝 Tahap 3/3: Mengisi konten Bab {current}/{total} — {title}...")

        filled = fill_rks_content(
            structure=structure,
            chunks=chunks,
            llm=llm,
            jenis_pekerjaan=jenis_pekerjaan,
            on_chapter_progress=on_progress,
            context_summary=context_summary
        )

        print("  ✅ Seluruh konten berhasil digenerate!")

        # Step 4: Save to draft
        _active_draft = RKSDraft()
        _active_draft.create(filled)

        # Step 5: Return summary
        summary = _active_draft.get_summary()
        draft_path = _active_draft.draft_path

        return (
            f"Draft RKS berhasil di-generate!\n\n"
            f"{summary}\n\n"
            f"📁 Draft tersimpan di: {draft_path}\n\n"
            f"Anda bisa:\n"
            f"- Meminta revisi pada bab tertentu (misal: 'revisi bab II, tambahkan ...')\n"
            f"- Meminta export ke .docx (misal: 'export ke docx')"
        )

    except Exception as e:
        error_msg = str(e)
        if "ERR_NGROK" in error_msg or "ngrok" in error_msg.lower():
            return (
                "⚠️ Koneksi ke server LLM terputus (tunnel ngrok offline/kedaluwarsa).\n"
                "Langkah perbaikan:\n"
                "1. Jalankan ulang server LLM + ngrok di Kaggle/Colab\n"
                "2. Salin URL ngrok baru ke config.py\n"
                "3. Restart agent ini"
            )
        return f"Error saat generate RKS: {error_msg}"


@tool
def revise_rks_section(
    nomor_bab: str,
    instruksi_revisi: str
) -> str:
    """Revisi satu bab dalam draft RKS yang aktif.
    Bisa mengubah konten, menambah/hapus sub-bab, atau mengubah format.
    Gunakan tool ini ketika user meminta perubahan pada bagian tertentu dari draft RKS.
    Parameter nomor_bab contoh: 'I', 'II', 'III', dst.
    """
    global _active_draft

    if not _active_draft.is_active():
        return "Tidak ada draft RKS aktif. Silakan generate draft terlebih dahulu menggunakan tool generate_rks_document."

    try:
        # Get current section
        current = _active_draft.get_section(nomor_bab)
        if not current:
            # Maybe user wants to add a new section
            if "tambah" in instruksi_revisi.lower() or "baru" in instruksi_revisi.lower():
                # Create new section via LLM
                llm = _get_llm()
                chunks = _search_multiple([instruksi_revisi])

                new_section = revise_section(
                    current_section={"nomor": nomor_bab, "judul": "", "sub_bab": []},
                    instruksi_revisi=instruksi_revisi,
                    chunks=chunks,
                    llm=llm
                )
                _active_draft.add_section(new_section)
                summary = _active_draft.get_summary()
                return f"Bab baru ditambahkan!\n\n{summary}"
            else:
                available_babs = [b.get("nomor", "?") for b in _active_draft.data.get("bab", [])]
                return f"Bab {nomor_bab} tidak ditemukan. Bab yang tersedia: {', '.join(available_babs)}"

        # Search for additional context if needed
        llm = _get_llm()
        chunks = _search_multiple([instruksi_revisi])

        # Revise via LLM
        revised = revise_section(
            current_section=current,
            instruksi_revisi=instruksi_revisi,
            chunks=chunks,
            llm=llm
        )

        # Update draft
        _active_draft.update_section(nomor_bab, revised)
        summary = _active_draft.get_summary()

        return (
            f"Bab {nomor_bab} berhasil direvisi!\n\n"
            f"{summary}\n\n"
            f"Ada revisi lain? Atau ketik 'export ke docx' untuk finalisasi."
        )

    except Exception as e:
        return f"Error saat revisi bab {nomor_bab}: {str(e)}"


@tool
def export_rks_to_docx(
    output_filename: str = ""
) -> str:
    """Export draft RKS yang aktif ke file .docx lengkap dengan cover page, 
    lembar pengesahan, daftar isi, dan konten bab dengan header/footer formal.
    Gunakan tool ini ketika user meminta export/unduh/simpan dokumen RKS ke format Word/docx.
    """
    global _active_draft

    if not _active_draft.is_active():
        return "Tidak ada draft RKS aktif. Silakan generate draft terlebih dahulu menggunakan tool generate_rks_document."

    try:
        draft_data = _active_draft.to_dict()
        output_path = render_to_docx(draft_data, output_filename)

        return (
            f"Dokumen RKS berhasil diekspor ke .docx!\n\n"
            f"📄 File: {output_path}\n\n"
            f"File mencakup:\n"
            f"- Halaman sampul (cover page) dengan border, logo, dan checkbox CSMS\n"
            f"- Lembar pengesahan\n"
            f"- Daftar isi (tekan F9 di Word untuk update)\n"
            f"- Konten bab lengkap dengan header/footer\n\n"
            f"Silakan buka file tersebut di Microsoft Word."
        )

    except Exception as e:
        return f"Error saat export ke docx: {str(e)}"


# List of tools to export to the agent module
rag_tools = [
    search_knowledge_base,
    generate_rks_document,
    revise_rks_section,
    export_rks_to_docx,
]