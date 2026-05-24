import streamlit as st
import torch
import gdown
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime

# ================================================
# KONFIGURASI HALAMAN
# ================================================
st.set_page_config(
    page_title="Mental Health AI Classifier",
    page_icon="🧠",
    layout="centered"
)

# ================================================
# KONSTANTA
# ================================================
MODEL_PATH = "model_hybrid_terbaik.pt"

# Ganti dengan ID file Google Drive kamu
# Cara dapat ID: klik kanan file di Drive → "Dapatkan link" → salin ID dari URL
# Contoh URL: https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWx/view
# ID-nya adalah: 1AbCdEfGhIjKlMnOpQrStUvWx
GDRIVE_FILE_ID = "https://drive.google.com/file/d/1FuWPSfBN9jgDvSJX3rN8X8jl1dvuslMA/view?usp=sharing"

LABEL_MAP = {
    0: "ADHD (Gangguan Fokus)",
    1: "Anxiety (Kecemasan)",
    2: "Bipolar (Perubahan Mood)",
    3: "Depression (Depresi)",
    4: "PTSD (Trauma)",
    5: "Kondisi Normal"
}

SARAN_MAP = {
    0: {
        "icon": "🎯",
        "warna": "#4A90D9",
        "saran": [
            "Gunakan teknik **Pomodoro**: kerja 25 menit, istirahat 5 menit",
            "Pecah tugas besar menjadi langkah-langkah kecil yang konkret",
            "Lakukan olahraga ringan sebelum memulai tugas kognitif berat"
        ]
    },
    1: {
        "icon": "🌬️",
        "warna": "#27AE60",
        "saran": [
            "Terapkan teknik pernapasan **4-7-8**: tarik 4 detik, tahan 7, hembuskan 8",
            "Gunakan teknik grounding **5-4-3-2-1** saat merasa cemas",
            "Hindari kafein dan stimulan sementara waktu"
        ]
    },
    2: {
        "icon": "⚖️",
        "warna": "#8E44AD",
        "saran": [
            "Jaga jadwal tidur yang **konsisten** setiap hari",
            "Catat perubahan mood harian dalam jurnal",
            "Hindari aktivitas berisiko tinggi saat periode energi berlebih"
        ]
    },
    3: {
        "icon": "☀️",
        "warna": "#E67E22",
        "saran": [
            "Mulai dengan **target mikro** yang sangat kecil (minum air, cuci muka)",
            "Jangkau satu orang terpercaya hari ini untuk berbagi perasaan",
            "Paparan sinar matahari pagi 10-15 menit untuk stabilisasi mood"
        ]
    },
    4: {
        "icon": "🛡️",
        "warna": "#C0392B",
        "saran": [
            "Gunakan **sensory grounding**: genggam benda bertekstur untuk hadir di sini",
            "Afirmasi: *'Saya aman sekarang, peristiwa itu telah berlalu'*",
            "Jauhi stimulus yang berpotensi memicu kilas balik"
        ]
    },
    5: {
        "icon": "✅",
        "warna": "#1ABC9C",
        "saran": [
            "Pertahankan pola tidur, nutrisi, dan aktivitas fisik yang seimbang",
            "Praktikkan **gratitude journaling** — catat 3 hal positif per hari",
            "Lanjutkan strategi koping adaptif yang sudah berjalan"
        ]
    }
}

# ================================================
# LOAD MODEL (dengan caching agar tidak reload)
# ================================================
@st.cache_resource
def load_model():
    """Download dan load model dari Google Drive sekali saja."""
    # Download model jika belum ada
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model dari Google Drive... (mungkin butuh 1-2 menit)"):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&confirm=t"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load model architecture + bobot
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=6
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict(text, model, tokenizer, device):
    """Fungsi inferensi: teks → label + confidence."""
    # Terjemahkan ke Bahasa Inggris jika perlu
    try:
        from deep_translator import GoogleTranslator
        text_en = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        text_en = text  # fallback ke teks asli

    inputs = tokenizer(
        text_en,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    pred_idx  = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    all_probs  = {LABEL_MAP[i]: round(probs[0][i].item() * 100, 2) for i in range(6)}

    return pred_idx, confidence, all_probs


# ================================================
# UI STREAMLIT
# ================================================
st.title("🧠 Mental Health AI Classifier")
st.markdown("**Sistem Skrining Kesehatan Mental berbasis NLP (RoBERTa Hybrid)**")
st.markdown("---")

# --- Disclaimer ---
st.warning(
    "⚠️ **Disclaimer:** Alat ini hanya untuk skrining awal dan edukasi. "
    "Bukan pengganti diagnosis medis profesional. "
    "Jika mengalami masalah kesehatan mental, segera hubungi profesional."
)

# --- Input Form ---
st.subheader("📋 Data Pasien")

col1, col2 = st.columns(2)
with col1:
    nama = st.text_input("Nama / Inisial", placeholder="Contoh: A atau Budi")
with col2:
    durasi = st.selectbox(
        "Durasi Gejala",
        ["< 2 Minggu (Akut)", "1–6 Bulan (Sub-akut)", "> 6 Bulan (Kronis)"]
    )

skala = st.slider("Skala Distres (1 = Ringan, 10 = Sangat Berat)", 1, 10, 5)

keluhan = st.text_area(
    "Ceritakan apa yang kamu rasakan:",
    placeholder="Contoh: Saya merasa sangat lelah dan tidak bersemangat melakukan apapun...",
    height=120
)

# --- Tombol Analisis ---
if st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True):
    if not keluhan.strip():
        st.error("Tolong isi kolom keluhan terlebih dahulu.")
    elif GDRIVE_FILE_ID == "ISI_DENGAN_ID_FILE_GOOGLE_DRIVE_KAMU":
        st.error("⚠️ GDRIVE_FILE_ID belum diisi! Ganti dengan ID file Google Drive kamu di baris kode `GDRIVE_FILE_ID`.")
    else:
        # Load model
        model, tokenizer, device = load_model()

        # Prediksi
        with st.spinner("🤖 Model sedang menganalisis..."):
            pred_idx, confidence, all_probs = predict(keluhan, model, tokenizer, device)

        label   = LABEL_MAP[pred_idx]
        info    = SARAN_MAP[pred_idx]
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

        # Tentukan level risiko
        if skala >= 8:
            risk_level = "🔴 RISIKO TINGGI"
            risk_color = "#E74C3C"
            risk_note  = "Disarankan segera berkonsultasi dengan psikolog/psikiater."
        elif skala >= 5:
            risk_level = "🟡 RISIKO SEDANG"
            risk_color = "#F39C12"
            risk_note  = "Pantau kondisi dan pertimbangkan konsultasi profesional."
        else:
            risk_level = "🟢 RISIKO RENDAH"
            risk_color = "#27AE60"
            risk_note  = "Lakukan observasi mandiri dan terapkan saran di bawah."

        st.markdown("---")
        st.subheader("📊 Hasil Analisis")

        # --- Hasil Utama ---
        st.markdown(
            f"""
            <div style="background:{info['warna']}20; border-left:4px solid {info['warna']};
                        padding:16px; border-radius:8px; margin-bottom:12px;">
              <h3 style="margin:0; color:{info['warna']}">
                {info['icon']} {label}
              </h3>
              <p style="margin:4px 0 0; color:#555; font-size:14px;">
                Confidence Score: <strong>{confidence*100:.1f}%</strong>
                &nbsp;|&nbsp; Timestamp: {timestamp}
                &nbsp;|&nbsp; Pasien: {nama or 'Anonim'}
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Level Risiko ---
        st.markdown(
            f"""
            <div style="background:{risk_color}15; border:1px solid {risk_color}40;
                        padding:10px 16px; border-radius:8px; margin-bottom:12px;">
              <strong style="color:{risk_color}">{risk_level}</strong>
              &nbsp;— Skala Distres: {skala}/10<br>
              <span style="font-size:13px; color:#666">{risk_note}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Rekomendasi ---
        st.markdown("**💡 Rekomendasi Intervensi Awal:**")
        for saran in info["saran"]:
            st.markdown(f"- {saran}")

        # --- Distribusi Probabilitas ---
        st.markdown("**📈 Distribusi Probabilitas Semua Kelas:**")
        st.bar_chart(all_probs)

        # --- Info Tambahan ---
        with st.expander("ℹ️ Informasi Anamnesis Lengkap"):
            st.markdown(f"""
            | Field | Nilai |
            |-------|-------|
            | Nama/Inisial | {nama or 'Anonim'} |
            | Durasi Gejala | {durasi} |
            | Skala Distres | {skala}/10 |
            | Hasil Prediksi | {label} |
            | Confidence | {confidence*100:.2f}% |
            | Timestamp | {timestamp} |
            """)

# ================================================
# FOOTER
# ================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "Proyek Data Mining · Model RoBERTa Hybrid · Dataset: Reddit Mental Health"
    "</p>",
    unsafe_allow_html=True
)
