import streamlit as st
import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime
import time

# ================================================
# KONFIGURASI HALAMAN
# ================================================
st.set_page_config(
    page_title="MindCheck AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================
# CUSTOM CSS - iOS STYLE MODERN
# ================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* iOS Card Style */
    .ios-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .ios-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Header Styles */
    .ios-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .ios-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .ios-subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
    }
    
    /* Button Styles */
    .ios-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .ios-button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Progress Ring */
    .progress-ring {
        position: relative;
        display: inline-block;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 1px solid #E5E7EB !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Badge */
    .ios-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ================================================
# KONSTANTA
# ================================================
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
        "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "saran": [
            "🎯 **Teknik Pomodoro**: Kerja 25 menit, istirahat 5 menit",
            "📝 **Task Breakdown**: Pecah tugas besar menjadi langkah-langkah kecil",
            "🏃 **Active Break**: Lakukan olahraga ringan sebelum tugas kognitif"
        ]
    },
    1: {
        "icon": "🌬️",
        "warna": "#27AE60",
        "gradient": "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)",
        "saran": [
            "🧘 **4-7-8 Breathing**: Tarik 4 detik, tahan 7, hembuskan 8",
            "🌍 **Grounding 5-4-3-2-1**: 5 lihat, 4 sentuh, 3 dengar, 2 cium, 1 rasa",
            "☕ **Reduce Stimulants**: Hindari kafein untuk sementara"
        ]
    },
    2: {
        "icon": "⚖️",
        "warna": "#8E44AD",
        "gradient": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
        "saran": [
            "⏰ **Sleep Schedule**: Jaga jadwal tidur konsisten setiap hari",
            "📓 **Mood Journal**: Catat perubahan mood harian",
            "⚡ **Energy Management**: Hindari aktivitas berisiko saat energi berlebih"
        ]
    },
    3: {
        "icon": "☀️",
        "warna": "#E67E22",
        "gradient": "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
        "saran": [
            "🎯 **Micro Goals**: Mulai dengan target sangat kecil (minum air, cuci muka)",
            "💬 **Social Connection**: Jangkau satu orang terpercaya hari ini",
            "🌞 **Morning Light**: Paparan sinar matahari pagi 10-15 menit"
        ]
    },
    4: {
        "icon": "🛡️",
        "warna": "#C0392B",
        "gradient": "linear-gradient(135deg, #f5af19 0%, #f12711 100%)",
        "saran": [
            "🖐️ **Sensory Grounding**: Genggam benda bertekstur untuk hadir di sini",
            "💪 **Safety Affirmation**: 'Saya aman sekarang, peristiwa itu telah berlalu'",
            "🚫 **Trigger Management**: Jauhi stimulus yang berpotensi memicu kilas balik"
        ]
    },
    5: {
        "icon": "✅",
        "warna": "#1ABC9C",
        "gradient": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
        "saran": [
            "⚖️ **Balanced Lifestyle**: Jaga pola tidur, nutrisi, dan aktivitas fisik",
            "📝 **Gratitude Journal**: Catat 3 hal positif setiap hari",
            "🌟 **Maintain Momentum**: Lanjutkan strategi koping adaptif yang sudah berjalan"
        ]
    }
}

# ================================================
# LOAD MODEL
# ================================================
@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with st.spinner("📥 Mengunduh model dari Hugging Face..."):
        model_path = hf_hub_download(
            repo_id="putrisadiyah19/mental-health-roberta",
            filename="model_hybrid_adaptive.pt"
        )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=6
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict(text, model, tokenizer, device):
    try:
        from deep_translator import GoogleTranslator
        text_en = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        text_en = text

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

    pred_idx   = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    all_probs  = {LABEL_MAP[i]: round(probs[0][i].item() * 100, 2) for i in range(6)}

    return pred_idx, confidence, all_probs

# ================================================
# UI STREAMLIT - iOS STYLE
# ================================================

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="ios-header">
        <h1 class="ios-title">🧠 MindCheck AI</h1>
        <p class="ios-subtitle">Skrining Kesehatan Mental Cerdas & Personal</p>
    </div>
    """, unsafe_allow_html=True)

# Main Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Disclaimer Card
with st.container():
    st.markdown("""
    <div class="ios-card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid #FF6B6B;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 24px;">⚠️</span>
            <div>
                <strong style="color: #FF6B6B;">Disclaimer Penting</strong><br>
                <span style="font-size: 0.9rem; color: #6B7280;">Alat ini hanya untuk skrining awal dan edukasi. Bukan pengganti diagnosis medis profesional. Jika mengalami masalah kesehatan mental, segera hubungi profesional.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Patient Data Card
st.markdown('<div class="ios-card">', unsafe_allow_html=True)
st.markdown("### 📋 Informasi Pasien")

col1, col2 = st.columns(2)
with col1:
    nama = st.text_input("Nama / Inisial", placeholder="Contoh: A. atau Budi", key="name_input")
    
    # Animated emoji picker
    mood = st.select_slider(
        "Bagaimana perasaanmu hari ini?",
        options=["😔 Sangat Buruk", "😕 Buruk", "😐 Biasa saja", "🙂 Baik", "😊 Sangat Baik"],
        value="😐 Biasa saja"
    )

with col2:
    durasi = st.selectbox(
        "Durasi Gejala",
        ["< 2 Minggu", "1–6 Bulan", "> 6 Bulan"],
        key="duration_select"
    )
    
    # Progress ring for distress scale
    st.markdown("**Skala Distres**")
    skala = st.slider("", 1, 10, 5, key="distress_slider")
    
    # Visual indicator for distress level
    if skala <= 3:
        st.markdown("<span style='color:#27AE60; font-size:0.9rem;'>✅ Level Rendah</span>", unsafe_allow_html=True)
    elif skala <= 7:
        st.markdown("<span style='color:#F39C12; font-size:0.9rem;'>⚠️ Level Sedang</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#E74C3C; font-size:0.9rem;'>🔴 Level Tinggi - Segera Konsultasi</span>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Keluhan Card
st.markdown('<div class="ios-card">', unsafe_allow_html=True)
st.markdown("### 💭 Ceritakan Pengalamanmu")
st.markdown("*Tuliskan apa yang kamu rasakan dengan jujur. Tidak ada penilaian, hanya pemahaman.*")

keluhan = st.text_area(
    "",
    placeholder="Contoh: Akhir-akhir ini saya merasa sangat lelah dan kehilangan semangat untuk melakukan aktivitas sehari-hari...",
    height=150,
    label_visibility="collapsed"
)

# Character counter
if keluhan:
    char_count = len(keluhan)
    st.markdown(f"<p style='text-align:right; font-size:0.8rem; color:#9CA3AF;'>{char_count} karakter</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Action Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary", key="analyze_btn")

# Analysis Result
if analyze_clicked:
    if not keluhan.strip():
        st.error("✨ Tolong isi keluhan terlebih dahulu sebelum menganalisis")
    else:
        model, tokenizer, device = load_model()

        # Loading animation
        with st.spinner("🤖 MindCheck AI sedang menganalisis..."):
            time.sleep(1)  # Simulated processing
            pred_idx, confidence, all_probs = predict(keluhan, model, tokenizer, device)

        label = LABEL_MAP[pred_idx]
        info = SARAN_MAP[pred_idx]
        timestamp = datetime.now().strftime("%d %B %Y, %H:%M")
        
        # Risk assessment
        if skala >= 8:
            risk_level = "🔴 Risiko Tinggi"
            risk_color = "#E74C3C"
            risk_note = "⚠️ Disarankan segera berkonsultasi dengan psikolog/psikiater"
        elif skala >= 5:
            risk_level = "🟡 Risiko Sedang"
            risk_color = "#F39C12"
            risk_note = "📊 Pantau kondisi dan pertimbangkan konsultasi profesional"
        else:
            risk_level = "🟢 Risiko Rendah"
            risk_color = "#27AE60"
            risk_note = "✅ Lakukan observasi mandiri dan terapkan saran di bawah"

        # Result Card
        st.markdown(f"""
        <div class="result-card" style="background: {info['gradient']}; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 48px;">{info['icon']}</span>
                    <h2 style="margin: 8px 0 0 0; color: white;">{label}</h2>
                </div>
                <div style="text-align: right;">
                    <div style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 12px;">
                        <span style="font-size: 0.9rem;">Confidence Score</span><br>
                        <strong style="font-size: 1.5rem;">{confidence*100:.1f}%</strong>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap;">
                <span class="ios-badge" style="background: rgba(255,255,255,0.2);">👤 {nama or 'Anonim'}</span>
                <span class="ios-badge" style="background: rgba(255,255,255,0.2);">📅 {timestamp}</span>
                <span class="ios-badge" style="background: rgba(255,255,255,0.2);">⏱️ {durasi}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk Level Card
        st.markdown(f"""
        <div class="ios-card" style="border-left: 4px solid {risk_color};">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h3 style="margin: 0; color: {risk_color};">{risk_level}</h3>
                    <p style="margin: 8px 0 0 0; color: #6B7280;">{risk_note}</p>
                </div>
                <div style="font-size: 36px;">
                    {'🔴' if skala >= 8 else '🟡' if skala >= 5 else '🟢'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations Card
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Rekomendasi Personal")
        st.markdown("*Berdasarkan analisis dan kondisi Anda, berikut saran yang dapat membantu:*")
        
        for saran in info["saran"]:
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 12px 16px; border-radius: 12px; margin: 8px 0;">
                {saran}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Probability Distribution
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Distribusi Probabilitas")
        st.markdown("*Analisis probabilitas untuk setiap kondisi kesehatan mental:*")
        
        # Sort probabilities for better visualization
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        for condition, prob in sorted_probs.items():
            st.markdown(f"""
            <div style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>{condition}</span>
                    <span><strong>{prob}%</strong></span>
                </div>
                <div style="background: #E5E7EB; border-radius: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: {prob}%; height: 8px; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Export option
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        st.markdown("### 📄 Laporan Lengkap")
        
        report = f"""
        **LAPORAN SKRINING MINDCHECK AI**
        ================================
        
        **Data Pasien:**
        - Nama/Inisial: {nama or 'Anonim'}
        - Tanggal: {timestamp}
        - Durasi Gejala: {durasi}
        - Skala Distres: {skala}/10 ({'Tinggi' if skala >= 8 else 'Sedang' if skala >= 5 else 'Rendah'})
        
        **Hasil Analisis:**
        - Diagnosis Prediksi: {label}
        - Confidence Score: {confidence*100:.1f}%
        - Mood Hari Ini: {mood}
        
        **Rekomendasi:**
        """
        for i, saran in enumerate(info["saran"], 1):
            report += f"\n{i}. {saran}"
        
        report += f"\n\n{risk_note}"
        
        st.download_button(
            label="📥 Download Laporan (TXT)",
            data=report,
            file_name=f"mindcheck_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, #ddd, transparent);">
    <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
        🧠 MindCheck AI — Didukung oleh RoBERTa Hybrid & Deep Learning<br>
        Proyek Data Mining untuk Kesehatan Mental
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)