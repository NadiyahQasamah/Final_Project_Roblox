import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle as pkl
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import plotly.express as px

# ===== KUSTOMISASI CSS (TIDAK BERUBAH) =====
st.set_page_config(
    page_title="💙 Sentiment AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CSS untuk tema biru yang menarik
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #7743DB !important;
    }
    
    .css-1d391kg .css-1d391kg {
        color: white;
    }
    
    /* Container utama */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Judul utama */
    h1 {
        color: #1e40af;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Sub judul */
    h2, h3 {
        color: #3b82f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Chat messages styling */
    .stChatMessage {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 15px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .css-1cpxqw2 {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px dashed #3b82f6;
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar title */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: white !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stRadio label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Metrics styling */
    .css-1r6slb0 {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        border-radius: 10px;
    }
    
    /* Info box */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
    }
    
    /* Chat input styling - memastikan masuk dalam container */
    .stChatInput {
        background: transparent !important;
        margin-bottom: 1rem;
    }
    .stChatInput > div {
        background-color: transparent !important; /* Background diubah jadi transparan */
        border-radius: 25px !important; /* Dibuat lebih melengkung */
        border: 2px solid #7743DB !important; /* Ditambahkan border ungu */
        box-shadow: none !important; /* Bayangan dihilangkan agar lebih bersih */
    }

    
    .stChatInput input {
        background: transparent !important;
        border: none !important;
        color: #1e40af !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stChatInput"] input {
        color: #333 !important;
        -webkit-text-fill-color: #333 !important;
    }
    /* Custom card styling */
    .custom-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #065f46;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
            
    .sentiment-neutral {
        background: linear-gradient(135deg, #FFF9C4 0%, #FFFACD 100%);
        color: #9E9D24;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ===== Muat model LSTM & Tokenizer (Tidak berubah) =====
@st.cache_resource
def load_assets():
    try:
        model = load_model("model_lstm.h5")
        with open("tokenizer_roblox.pkl", "rb") as f:
            tokenizer = pkl.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Gagal memuat aset model: {e}. Pastikan 'model_lstm.h5' dan 'tokenizer_roblox.pkl' ada di folder yang sama.")
        return None, None

lstm_model, tokenizer = load_assets()

# ===== MODIFIKASI: Fungsi prediksi untuk 3 KELAS =====
def predict_sentiment_lstm(text):
    """Fungsi ini menggunakan model LSTM untuk prediksi 3 kelas (Positif, Negatif, Netral)."""
    if not lstm_model or not tokenizer:
        return "Netral", 0.5
    
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    
    # === BAGIAN PENTING UNTUK DIAGNOSIS ===
    prediction_result = lstm_model.predict(padded)
    print("-----------------------------------------")
    print("DIAGNOSIS OUTPUT MODEL:")
    print("BENTUK OUTPUT (SHAPE):", prediction_result.shape)
    print("ISI OUTPUT (KONTEN):", prediction_result[0])
    print("-----------------------------------------")
    # =======================================
    
    # Logika untuk 3 kelas (seperti sebelumnya)
    pred_probs = prediction_result[0]
    pred_class_index = np.argmax(pred_probs)
    confidence = float(pred_probs[pred_class_index])
    class_labels = ["Negatif", "Netral", "Positif"] # Asumsi urutan 0, 1, 2
    sentiment = class_labels[pred_class_index]
        
    return sentiment, confidence

# ===== Fungsi generate penjelasan (Tidak berubah) =====
def generate_explanation_with_api(text, sentiment, confidence):
    """Meminta penjelasan dari Gemini berdasarkan hasil analisis LSTM."""
    try:
        explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Anda adalah asisten AI. Sebuah model LSTM lokal telah menganalisis teks berikut dan menyimpulkan sentimennya adalah '{sentiment}' dengan tingkat kepercayaan {confidence:.2%}.
            Tugas Anda adalah memberikan penjelasan mengapa teks tersebut bisa dianggap memiliki sentimen '{sentiment}'. Jangan analisis ulang sentimennya, cukup jelaskan hasil yang sudah ada.
            """),
            ("human", "Teks: {text}")
        ])
        explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)
        response = explanation_chain.run({"text": text})
        return response.strip()
    except Exception as e:
        return f"Tidak bisa mendapatkan penjelasan dari AI: {e}"

# ===== Header, Sidebar, Konfigurasi LLM (Tidak berubah) =====
# ... (Kode dari Header hingga Konfigurasi LLM tetap sama) ...
st.markdown("""<div style="text-align: center; padding: 2rem 0;"><h1 style="font-size: 3rem; background: linear-gradient(135deg, #1e40af, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">🤖 Sentiment AI Chat</h1><p style="color: #64748b; font-size: 1.2rem; margin: 0;">Analisis Sentimen Cerdas dengan AI 🚀</p></div>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("""<div style="text-align: center; padding: 1rem 0;"><h2 style="color: white; margin-bottom: 1rem;">🧭 Navigasi</h2></div>""", unsafe_allow_html=True)
    menu = st.radio("Pilih halaman:", ["💬 Chat & Analisis", "ℹ️ Tentang Aplikasi"], label_visibility="collapsed")
api_key = "AIzaSyApZtVEV_jCDrP1Rk9PdUSGgYli0NKlIwM"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7, 
    google_api_key=api_key)
prompt_template = ChatPromptTemplate.from_messages([("system", "Anda adalah asisten AI yang ramah dan membantu. Jawab pertanyaan pengguna dengan jelas seputar analisis sentimen."), ("human", "Pertanyaan: {question}\nRiwayat chat:\n{history}")])
chain = LLMChain(llm=llm, prompt=prompt_template)


# ===== KONTEN HALAMAN BERDASARKAN MENU =====
if menu == "💬 Chat & Analisis":
    st.markdown("""<div class="custom-card"><h3 style="color: #1e40af; margin-bottom: 1rem;">📁 Upload File untuk Analisis</h3><p style="color: #64748b;">Upload file CSV atau PDF untuk analisis sentimen otomatis menggunakan model LSTM lokal.</p></div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file di sini", type=["csv", "pdf"], help="Pilih file CSV dengan kolom 'text' atau file PDF untuk dianalisis", label_visibility="collapsed")
    
    if uploaded_file and (lstm_model and tokenizer):
        with st.spinner('🔄 Menganalisis file dengan model LSTM lokal...'):
            texts = []
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                if "text" not in df.columns: st.error("❌ CSV harus memiliki kolom bernama 'text'.")
                else: texts = df["text"].astype(str).tolist()
            elif uploaded_file.type == "application/pdf":
                pdf = PdfReader(uploaded_file)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: texts.append(page_text)

            if texts:
                results = []
                progress_bar = st.progress(0)
                for i, text in enumerate(texts):
                    sentiment, confidence = predict_sentiment_lstm(text)
                    explanation = "Penjelasan dihasilkan saat analisis individu." 
                    results.append((sentiment, confidence, explanation))
                    progress_bar.progress((i + 1) / len(texts))
                
                st.markdown("### 📊 Hasil Analisis Sentimen (by LSTM Model)")
                positive_count = sum(1 for r in results if r[0] == "Positif")
                negative_count = sum(1 for r in results if r[0] == "Negatif")
                # MODIFIKASI: Menghitung kelas Netral secara eksplisit
                neutral_count = sum(1 for r in results if r[0] == "Netral")

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Teks", len(results))
                with col2: st.metric("Positif", positive_count)
                with col3: st.metric("Negatif", negative_count)
                with col4: st.metric("Netral", neutral_count)

                if len(results) > 1:
                    # MODIFIKASI: Menambahkan data Netral ke pie chart
                    fig = px.pie(
                        values=[positive_count, negative_count, neutral_count],
                        names=["Positif", "Negatif", "Netral"],
                        title="Distribusi Sentimen",
                        color_discrete_sequence=["#10b981", "#ef4444", "#FBC02D"] # Hijau, Merah, Kuning
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📝 Detail Hasil")
                for i, (text, (sentiment, confidence, explanation)) in enumerate(zip(texts, results)):
                    # MODIFIKASI: Menambahkan logika untuk kelas Netral
                    if sentiment == "Positif":
                        sentiment_class = "sentiment-positive"
                    elif sentiment == "Negatif":
                        sentiment_class = "sentiment-negative"
                    else: # Netral
                        sentiment_class = "sentiment-neutral"
                        
                    st.markdown(f"""
                    <div class="custom-card">
                        <div class="{sentiment_class}">{sentiment} ({confidence:.2%} confidence)</div>
                        <p style="margin-top: 0.5rem; color: #6b7280; font-style: italic;">{text[:200]}{'...' if len(text) > 200 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat Section
    st.markdown("""<div class="custom-card">
                <h3 style="color: #1e40af; margin-bottom: 1rem;">
                💬 Chat dengan AI</h3><p style="color: #64748b;">
                Tanyakan apapun dan dapatkan analisis sentimen real-time</p>
    </div>""", unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_input := st.chat_input("💭 Tulis pesan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Menganalisis sentimen & menyiapkan jawaban..."):
                sentiment, confidence = predict_sentiment_lstm(user_input)
                explanation = generate_explanation_with_api(user_input, sentiment, confidence)
                history_text = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages[-4:]]) or " "
                response_text = chain.run({"question": user_input, "history": history_text})
                
                # MODIFIKASI: Menambahkan logika untuk kelas Netral
                if sentiment == "Positif":
                    sentiment_class = "sentiment-positive"
                elif sentiment == "Negatif":
                    sentiment_class = "sentiment-negative"
                else: # Netral
                    sentiment_class = "sentiment-neutral"

                st.markdown(f"""
                <div class="{sentiment_class}" style="margin-bottom: 1rem;">
                    <b>Sentimen (LSTM):</b> {sentiment} ({confidence:.2%})<br>
                    <b>Penjelasan (AI):</b> {explanation}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(response_text)
                
                full_response = f"""**Sentimen:** {sentiment} ({confidence:.2%})\n\n**Penjelasan:** {explanation}\n\n{response_text}"""
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# ===== MENU TENTANG APLIKASI =====
elif menu == "ℹ️ Tentang Aplikasi":
    
    st.markdown("""
     <h2 style="color: #1e40af; text-align: center; margin-bottom: 2rem;">
        🚀 Tentang Sentiment AI Chat
    </h2>
    
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #64748b; line-height: 1.6;">
            Aplikasi analisis sentimen berbasis AI yang menggabungkan kekuatan 
            <strong>LSTM Deep Learning</strong> dengan <strong>Google Generative AI</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #3b82f6; margin-bottom: 1rem;">✨ Fitur Utama</h3>
            <ul style="color: #374151; line-height: 2;">
                <li><strong>Chat Real-time</strong> - Interaksi langsung dengan AI</li>
                <li><strong>Hybrid Analysis</strong> - LSTM Model + Google Gemini AI</li>
                <li><strong>Upload File</strong> - Support CSV dan PDF</li>
                <li><strong>Visualisasi Data</strong> - Chart dan statistik interaktif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #3b82f6; margin-bottom: 1rem;">🛠️ Teknologi</h3>
            <ul style="color: #374151; line-height: 2;">
                <li><strong>Streamlit</strong> - Web framework</li>
                <li><strong>TensorFlow/Keras</strong> - LSTM model (loaded)</li>
                <li><strong>Google Gemini AI</strong> - Sentiment analysis engine</li>
                <li><strong>Plotly</strong> - Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics or additional info
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #3b82f6; margin-bottom: 1rem;">📈 Model Performance</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 10px;">
                <h4 style="color: #1e40af; margin: 0;">LSTM + Gemini</h4>
                <p style="color: #64748b; margin: 0.5rem 0;">Hybrid Analysis</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 10px;">
                <h4 style="color: #1e40af; margin: 0;">Real-time</h4>
                <p style="color: #64748b; margin: 0.5rem 0;">Instant Analysis</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 10px;">
                <h4 style="color: #1e40af; margin: 0;">Multi-format</h4>
                <p style="color: #64748b; margin: 0.5rem 0;">CSV & PDF Support</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #1e40af, #3b82f6); border-radius: 15px; color: white;">
        <h3 style="margin-bottom: 1rem; color: white;">🎉 Selamat Menggunakan!</h3>
        <p style="margin: 0; opacity: 0.9;">
            Dibuat dengan ❤️ menggunakan Streamlit dan AI terdepan
        </p>
    </div>
    """, unsafe_allow_html=True)
        # Footer dalam container
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem 0; color: #64748b; border-top: 1px solid #e2e8f0;">
        <p style="margin: 0; font-size: 0.9rem;">
            Powered by 🤖 <strong>Sentiment AI</strong> | Built with Streamlit 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)
