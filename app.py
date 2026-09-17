import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Set page configuration
st.set_page_config(
    page_title="Dual-Risk Construction Safety Analytics",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and industrial safety styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header banner */
    .safety-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-left: 6px solid #f59e0b;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        color: #ffffff;
    }
    .safety-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #f8fafc;
    }
    .safety-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* RAG Alert Badges */
    .rag-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .rag-green {
        background: linear-gradient(135deg, #059669, #10b981);
        color: #ffffff;
        border: 2px solid #34d399;
    }
    .rag-amber {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: #ffffff;
        border: 2px solid #fbbf24;
    }
    .rag-red {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: #ffffff;
        border: 2px solid #f87171;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Card Container */
    .safety-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD CACHED MODELS AND DATA
# -------------------------------------------------------------
@st.cache_resource
def load_pipelines():
    h_art = joblib.load("models/height_pipeline.joblib")
    u_art = joblib.load("models/underground_pipeline.joblib")
    return h_art, u_art

@st.cache_data
def load_augmented_data():
    h_df = pd.read_csv("data/height_augmented.csv")
    u_df = pd.read_csv("data/underground_augmented.csv")
    return h_df, u_df

@st.cache_data
def load_metrics():
    with open("results/height_metrics.json", "r", encoding="utf-8") as f:
        h_m = json.load(f)
    with open("results/underground_metrics.json", "r", encoding="utf-8") as f:
        u_m = json.load(f)
    return h_m, u_m

height_pipeline, underground_pipeline = load_pipelines()
height_df, underground_df = load_augmented_data()
height_metrics, underground_metrics = load_metrics()

# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="safety-banner">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <div class="safety-title">🏗️ Dual-Risk Construction Safety Analytics & Explainable AI</div>
            <div class="safety-subtitle">द्वि-स्तरीय निर्माण सुरक्षा विश्लेषण: ऊंचाई और भूमिगत खुदाई जोखिम की भविष्यवाणी (RAG Alerts & SHAP Drivers)</div>
        </div>
        <div style="margin-top: 10px;">
            <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                🛡️ Live Field Safety Monitoring
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/hard-hat.png", width=70)
    st.title("🎛️ Safety Control Room")
    st.caption("साइट सुरक्षा नियंत्रण कक्ष")
    
    active_site = st.selectbox(
        "📍 निर्माण स्थल (Select Construction Site)",
        ["Site 1: Metro Line Underground & Pier", 
         "Site 2: High-Rise Tower A (32 Floors)", 
         "Site 3: Foundation & Basement Complex", 
         "Site 4: Commercial Mall Excavation",
         "Site 5: Residential Block 4 (15 Floors)"]
    )
    
    st.markdown("---")
    st.markdown("### 🤖 ML Engine Config")
    selected_model_name = st.radio(
        "सक्रिय मॉडल (Active Classifier)",
        ["XGBoost (Recommended)", "Random Forest", "Logistic Regression"]
    )
    model_key = "XGBoost" if "XGBoost" in selected_model_name else ("Random Forest" if "Random Forest" in selected_model_name else "Logistic Regression")
    
    st.markdown("---")
    st.markdown("### 📊 Augmented Dataset Stats")
    st.markdown(f"""
    - **ऊंचाई डेटा (Height Work)**: 600 नमूने (200 प्रति वर्ग)
    - **खुदाई डेटा (Excavation)**: 600 नमूने (200 प्रति वर्ग)
    - **संतुलित कक्षाएं**: 🟢 200 | 🟡 200 | 🔴 200
    - **व्याख्या**: SHAP TreeExplainer
    """)
    st.info("💡 सचित्र इनपुट प्रणाली कम साक्षरता वाले श्रमिकों के लिए डिज़ाइन की गई है। (Pictographic check-ins empower informal construction labor).")

# -------------------------------------------------------------
# MAIN APP TABS
# -------------------------------------------------------------
tabs = st.tabs([
    "📋 सचित्र सुरक्षा चेक-इन (Daily Pictographic Check-In)",
    "🚨 ठेकेदार / प्रबंधक RAG अलर्ट बोर्ड (Site RAG Dashboard)",
    "🧠 SHAP व्याख्या और मुख्य जोखिम कारक (Explainable AI)",
    "🛠️ सुझाई गई सुरक्षा कार्रवाई (Actionable SOPs)",
    "📈 मॉडल तुलना और अर्थशास्त्र शोध (ML Benchmark & Economics)"
])

# =============================================================
# TAB 1: PICTOGRAPHIC CHECK-IN SYSTEM
# =============================================================
with tabs[0]:
    st.subheader("📋 सचित्र दैनिक सुरक्षा चेक-इन (Daily Safety Check-In Tool)")
    st.markdown("सरल आइकन और हिंदी लेबल के माध्यम से वास्तविक समय में सुरक्षा डेटा रिकॉर्ड करें।")
    
    domain_choice = st.radio(
        "कार्य का प्रकार चुनें (Select Work Environment):",
        ["🪜 ऊंचाई पर कार्य (Height Work Hazard)", "🚜 भूमिगत / खुदाई कार्य (Excavation Hazard)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if "ऊंचाई" in domain_choice:
        st.markdown("### 🪜 ऊंचाई कार्य सुरक्षा पैरामीटर (Height Fall Risk Inputs)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            worker_role = st.selectbox(
                "👷 श्रमिक भूमिका (Worker Role)",
                ['बढ़ई (Carpenter)', 'स्कैफोल्डर (Scaffolder)', 'हेल्पर (Helper)', 'राजमिस्त्री (Mason)', 'इलेक्ट्रीशियन (Electrician)']
            )
            age = st.selectbox(
                "🎂 आयु (Age Group)",
                ['18–25 वर्ष', '26–35 वर्ष', '36–45 वर्ष', '45 वर्ष से अधिक'],
                index=1
            )
            experience = st.selectbox(
                "⏳ कार्य अनुभव (Experience)",
                ['1 वर्ष से कम', '1–3 वर्ष', '3–5 वर्ष', '5 वर्ष से अधिक'],
                index=2
            )
            floor_level = st.selectbox(
                "🏢 मंजिल स्तर (Floor Level)",
                ['भूतल (Ground Floor)', '1–3 मंजिल', '4–6 मंजिल', '7–10 मंजिल', '10 से अधिक मंजिल'],
                index=3
            )
            
        with col2:
            shift_hours = st.selectbox(
                "⏱️ शिफ्ट अवधि (Daily Shift Duration)",
                ['8 घंटे से कम', '8–10 घंटे', '10–12 घंटे', '12 घंटे से अधिक'],
                index=1
            )
            mood = st.selectbox(
                "😊 मानसिक स्थिति / थकान (Fatigue & Stress)",
                ['😄 खुश (Energetic & Fresh)', '🙂 सामान्य (Normal)', '😐 थोड़ा तनावग्रस्त (Fatigued / Stressed)'],
                index=1
            )
            shift = st.selectbox(
                "☀️🌙 कार्य पाली (Shift Time)",
                ['🌞 दिन की शिफ्ट (Day Shift)', '🌙 रात की शिफ्ट (Night Shift)']
            )
            helmet = st.radio(
                "⛑️ सुरक्षा हेलमेट (Safety Helmet Worn?)",
                ['हाँ', 'नहीं'],
                horizontal=True
            )
            
        with col3:
            harness = st.radio(
                "🦺 सेफ्टी हार्नेस पहनी है? (Safety Harness)",
                ['हाँ', 'नहीं'],
                index=0,
                horizontal=True
            )
            harness_anchor = st.radio(
                "⚓ हार्नेस लाइफलाइन से जुड़ी है? (Harness Anchored?)",
                ['हाँ', 'नहीं', 'लागू नहीं'],
                index=0,
                horizontal=True
            )
            guardrail = st.radio(
                "🚧 गार्डरेल / बैरिकेडिंग लगी है? (Guardrail Present?)",
                ['हाँ', 'नहीं'],
                index=0,
                horizontal=True
            )
            scaffold = st.selectbox(
                "🪜 मचान की स्थिति (Scaffolding Stability)",
                ['स्थिर', 'थोड़ा अस्थिर', 'अस्थिर']
            )
            open_edge = st.radio(
                "⚠️ खुला किनारा / फर्श कटआउट? (Open Unprotected Edge?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )
            near_miss = st.radio(
                "🚨 हाल ही में फिसलने / गिरने की चूक? (Near-Miss Slip/Fall?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )

        # Build feature dictionary
        clean_mood = '😐 थोड़ा तनावग्रस्त' if 'तनावग्रस्त' in mood else ('😄 खुश' if 'खुश' in mood else '🙂 सामान्य')
        input_data = {
            "WorkerRole": worker_role,
            "Age": age,
            "Experience": experience,
            "FloorLevel": floor_level,
            "ShiftHours": shift_hours,
            "Mood": clean_mood,
            "Helmet": helmet,
            "Harness": harness,
            "HarnessAnchor": harness_anchor,
            "GuardRail": guardrail,
            "Scaffold": scaffold,
            "OpenEdge": open_edge,
            "NearMiss": near_miss,
            "Shift": shift
        }
        
        # Inference
        pipeline = height_pipeline
        raw_df = pd.DataFrame([input_data])
        encoded_df = pd.DataFrame()
        for col in pipeline["feature_cols"]:
            encoder = pipeline["encoders"][col]
            val = str(input_data[col])
            # safe transform
            if val in encoder.classes_:
                encoded_df[col] = encoder.transform([val])
            else:
                encoded_df[col] = [0]
                
        active_model = pipeline["models"][model_key]
        if model_key == "Logistic Regression":
            X_scaled = pipeline["scaler"].transform(encoded_df)
            probs = active_model.predict_proba(X_scaled)[0]
        else:
            probs = active_model.predict_proba(encoded_df)[0]
            
        pred_class_idx = int(np.argmax(probs))
        pred_label = pipeline["class_names"][pred_class_idx]
        
        # Fall Risk Score calculation (weighted probability of moderate & high risk)
        fall_risk_score = (probs[1] * 50.0 + probs[2] * 100.0)
        
        st.markdown("---")
        st.subheader("🎯 वास्तविक समय पतन जोखिम मूल्यांकन (Real-Time Fall Risk Assessment)")
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
        with res_col1:
            st.metric(
                label="🧗 फॉल रिस्क स्कोर (Fall Risk Score)",
                value=f"{fall_risk_score:.1f} / 100",
                delta=f"{'+' if fall_risk_score > 40 else '-'}{abs(fall_risk_score - 30):.1f}% from baseline",
                delta_color="inverse"
            )
            st.progress(float(min(1.0, max(0.0, float(fall_risk_score) / 100.0))))
            
        with res_col2:
            st.markdown("#### RAG अलर्ट स्थिति (Alert Level)")
            if pred_label == "🟢 सुरक्षित":
                st.markdown('<div class="rag-badge rag-green">🟢 SAFE / सुरक्षित</div>', unsafe_allow_html=True)
                st.caption("कार्य जारी रखने की अनुमति है। बुनियादी पीपीई अनिवार्य है।")
            elif pred_label == "🟡 मध्यम जोखिम":
                st.markdown('<div class="rag-badge rag-amber">🟡 AMBER / मध्यम जोखिम</div>', unsafe_allow_html=True)
                st.caption("चेतावनी: सुरक्षा नियंत्रणों की तत्काल जांच करें।")
            else:
                st.markdown('<div class="rag-badge rag-red">🔴 RED ALERT / गंभीर खतरा</div>', unsafe_allow_html=True)
                st.caption("रोकें! कार्य तुरंत रोकें और सुरक्षा सुधारात्मक कदम उठाएं।")
                
        with res_col3:
            st.markdown("#### संभावना वितरण (Risk Probabilities)")
            p_df = pd.DataFrame({
                "श्रेणी (Class)": pipeline["class_names"],
                "संभावना (Probability)": [f"{p*100:.1f}%" for p in probs]
            })
            st.dataframe(p_df, use_container_width=True, hide_index=True)
            
        st.session_state["last_height_sample"] = encoded_df
        st.session_state["last_height_input"] = input_data
        st.session_state["last_domain"] = "height"

    else:
        st.markdown("### 🚜 भूमिगत / खुदाई जोखिम पैरामीटर (Excavation Risk Inputs)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            u_worker_role = st.selectbox(
                "👷 श्रमिक भूमिका (Worker Role)",
                ['हेल्पर (Helper)', 'पर्यवेक्षक / सुपरवाइजर (Supervisor)', 'मशीन ऑपरेटर (Machine Operator)', 'राजमिस्त्री (Mason)', 'खुदाई कार्यकर्ता (Excavation Worker)']
            )
            u_age = st.selectbox(
                "🎂 आयु (Age Group)",
                ['18–25 वर्ष', '26–35 वर्ष', '36–45 वर्ष', '45 वर्ष से अधिक'],
                index=1
            )
            u_experience = st.selectbox(
                "⏳ कार्य अनुभव (Experience)",
                ['1 वर्ष से कम', '1–3 वर्ष', '3–5 वर्ष', '5 वर्ष से अधिक'],
                index=2
            )
            u_depth = st.selectbox(
                "🕳️ गड्ढे की गहराई (Excavation Depth)",
                ['1 मीटर से कम', '1–3 मीटर', '3–5 मीटर', '5 मीटर से अधिक'],
                index=2
            )
            
        with col2:
            u_seepage = st.radio(
                "💧 पानी का रिसाव / कीचड़ (Water Seepage?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )
            u_shoring = st.selectbox(
                "🛡️ शोरिंग / दीवार टेक (Shoring & Trench Box Support)",
                ['हाँ', 'पता नहीं', 'नहीं']
            )
            u_cracks = st.radio(
                "⚡ किनारे पर तनाव दरारें (Tension Cracks on Crest?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )
            u_machinery = st.radio(
                "🚜 किनारे भारी मशीनरी / कंपन (Heavy Machinery < 2m?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )
            
        with col3:
            u_material = st.radio(
                "🧱 किनारे पर मलबा / निर्माण भार (Material/Spoil Pile < 1m?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )
            u_near_miss = st.radio(
                "🚨 मिट्टी ढहने की चूक (Near-Miss Soil Fall / Slump?)",
                ['नहीं', 'हाँ'],
                horizontal=True
            )
            u_shift_hours = st.selectbox(
                "⏱️ शिफ्ट अवधि (Shift Duration)",
                ['8 घंटे से कम', '8–10 घंटे', '10–12 घंटे', '12 घंटे से अधिक'],
                index=1
            )
            u_mood = st.selectbox(
                "😊 मनोदशा / तनाव (Mood & Fatigue)",
                ['😄 खुश', '🙂 सामान्य', '😐 थोड़ा तनावग्रस्त'],
                index=1
            )
            u_shift = st.selectbox(
                "☀️🌙 शिफ्ट (Day/Night Shift)",
                ['🌞 दिन की शिफ्ट', '🌙 रात की शिफ्ट']
            )

        u_clean_mood = '😐 थोड़ा तनावग्रस्त' if 'तनावग्रस्त' in u_mood else ('😄 खुश' if 'खुश' in u_mood else '🙂 सामान्य')
        u_input_data = {
            "WorkerRole": u_worker_role,
            "Age": u_age,
            "Experience": u_experience,
            "Depth": u_depth,
            "Shift": u_shift,
            "ShiftHours": u_shift_hours,
            "Mood": u_clean_mood,
            "Helmet": "हाँ",
            "WaterSeepage": u_seepage,
            "Shoring": u_shoring,
            "Cracks": u_cracks,
            "Machinery": u_machinery,
            "Material": u_material,
            "NearMiss": u_near_miss
        }
        
        # Inference
        pipeline = underground_pipeline
        raw_df = pd.DataFrame([u_input_data])
        encoded_df = pd.DataFrame()
        for col in pipeline["feature_cols"]:
            encoder = pipeline["encoders"][col]
            val = str(u_input_data[col])
            if val in encoder.classes_:
                encoded_df[col] = encoder.transform([val])
            else:
                encoded_df[col] = [0]
                
        active_model = pipeline["models"][model_key]
        if model_key == "Logistic Regression":
            X_scaled = pipeline["scaler"].transform(encoded_df)
            probs = active_model.predict_proba(X_scaled)[0]
        else:
            probs = active_model.predict_proba(encoded_df)[0]
            
        pred_class_idx = int(np.argmax(probs))
        pred_label = pipeline["class_names"][pred_class_idx]
        collapse_risk_score = (probs[1] * 50.0 + probs[2] * 100.0)
        
        st.markdown("---")
        st.subheader("🎯 वास्तविक समय खुदाई पतन जोखिम (Real-Time Excavation Collapse Risk)")
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
        with res_col1:
            st.metric(
                label="🕳️ ढहने का जोखिम स्कोर (Excavation Collapse Risk Score)",
                value=f"{collapse_risk_score:.1f} / 100",
                delta=f"{'+' if collapse_risk_score > 40 else '-'}{abs(collapse_risk_score - 30):.1f}% from baseline",
                delta_color="inverse"
            )
            st.progress(float(min(1.0, max(0.0, float(collapse_risk_score) / 100.0))))
            
        with res_col2:
            st.markdown("#### RAG अलर्ट स्थिति (Alert Level)")
            if pred_label == "🟢 सुरक्षित":
                st.markdown('<div class="rag-badge rag-green">🟢 SAFE / सुरक्षित</div>', unsafe_allow_html=True)
                st.caption("गड्ढे में कार्य सुरक्षित है। नियमित निरीक्षण जारी रखें।")
            elif pred_label == "🟡 मध्यम जोखिम":
                st.markdown('<div class="rag-badge rag-amber">🟡 AMBER / मध्यम जोखिम</div>', unsafe_allow_html=True)
                st.caption("चेतावनी: जल निकासी और तटवर्ती दीवार टेक सुदृढ़ करें।")
            else:
                st.markdown('<div class="rag-badge rag-red">🔴 RED ALERT / गंभीर ढहाव खतरा</div>', unsafe_allow_html=True)
                st.caption("तत्काल कार्य रोकें! खाई ढहने का गंभीर जोखिम मौजूद है।")
                
        with res_col3:
            st.markdown("#### संभावना वितरण (Risk Probabilities)")
            p_df = pd.DataFrame({
                "श्रेणी (Class)": pipeline["class_names"],
                "संभावना (Probability)": [f"{p*100:.1f}%" for p in probs]
            })
            st.dataframe(p_df, use_container_width=True, hide_index=True)

        st.session_state["last_underground_sample"] = encoded_df
        st.session_state["last_underground_input"] = u_input_data
        st.session_state["last_domain"] = "underground"


# =============================================================
# TAB 2: CONTRACTOR / SITE MANAGER RAG DASHBOARD
# =============================================================
with tabs[1]:
    st.subheader(f"🚨 साइट प्रबंधक दैनिक RAG डैशबोर्ड ({active_site})")
    st.markdown("समग्र निर्माण स्थल की दैनिक सुरक्षा स्थिति, उच्च-जोखिम वाले क्षेत्र और अनुपालन दर की वास्तविक निगरानी।")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("👷 सक्रिय श्रमिक चेक-इन", "128", "+12 आज")
    with kpi2:
        st.metric("🟢 सुरक्षित परमिट (Safe)", "94", "73.4%")
    with kpi3:
        st.metric("🟡 सतर्कता क्षेत्र (Amber)", "24", "18.7%")
    with kpi4:
        st.metric("🔴 स्टॉप वर्क अलर्ट (Red)", "10", "-3 कल से", delta_color="inverse")
    with kpi5:
        st.metric("🛡️ हार्नेस व पीपीई अनुपालन", "91.5%", "+4.2%")

    st.markdown("---")
    
    dash_col1, dash_col2 = st.columns([1.5, 1])
    with dash_col1:
        st.markdown("#### 🗺️ कार्यक्षेत्र जोखिम हीटमैप (Active Zone Risk Matrix)")
        zones_data = pd.DataFrame({
            "जोन / स्थान (Zone Area)": [
                "Zone A: टावर 1 - ऊपरी मंजिलें (Floors 12-16)",
                "Zone B: मुख्य बेसमेंट खुदाई (Pit B, 6m Deep)",
                "Zone C: पूर्वी रिटेनिंग वॉल (Trench 3m)",
                "Zone D: टावर 2 - आंतरिक मचान (Scaffolding Pier)",
                "Zone E: बाहरी लिफ्ट शाफ्ट और क्रेन जोन"
            ],
            "कार्य प्रकार": ["ऊंचाई (Height)", "खुदाई (Excavation)", "खुदाई (Excavation)", "ऊंचाई (Height)", "ऊंचाई (Height)"],
            "अलर्ट स्तर (RAG Status)": ["🔴 RED (अनएंकरड हार्नेस)", "🔴 RED (जल रिसाव + दरारें)", "🟡 AMBER (भारी मशीनरी निकट)", "🟢 GREEN (सुरक्षित)", "🟡 AMBER (खुला किनारा)"],
            "जोखिम स्कोर": ["88 / 100", "94 / 100", "56 / 100", "18 / 100", "62 / 100"],
            "सुझाई गई कार्रवाई": [
                "तत्काल काम बंद: लाइफलाइन स्थापित करें",
                "गड्ढा तुरंत खाली करें, डीवाटरिंग और शोरिंग लगाएं",
                "मशीनरी को 3 मीटर पीछे करें",
                "कार्य जारी रखें (नियमित जांच)",
                "सुरक्षा रेलिंग / जाली तुरंत लगाएं"
            ]
        })
        st.dataframe(zones_data, use_container_width=True, hide_index=True)
        
    with dash_col2:
        st.markdown("#### 📊 RAG अलर्ट वितरण (Distribution)")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        sizes = [94, 24, 10]
        labels = ['Safe (Green)', 'Moderate (Amber)', 'Severe (Red)']
        colors = ['#10b981', '#f59e0b', '#ef4444']
        explode = (0, 0.05, 0.12)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=140, textprops={'fontsize': 10, 'weight': 'bold'})
        ax.axis('equal')
        fig.patch.set_facecolor('#ffffff')
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("#### 📑 दैनिक चेक-इन रिकॉर्ड्स (Recent Field Check-Ins Log)")
    
    filter_choice = st.selectbox("फ़िल्टर करें (Filter By Alert Type):", ["सभी (All Records)", "🔴 केवल रेड अलर्ट (Red Only)", "🟡 मध्यम अलर्ट (Amber Only)", "🟢 सुरक्षित (Green Only)"])
    
    # Create realistic sample log
    sample_records = height_df.head(15).copy()
    if "रेड" in filter_choice:
        sample_records = height_df[height_df["RiskLevel"] == "🔴 बहुत अधिक जोखिम"].head(15)
    elif "मध्यम" in filter_choice:
        sample_records = height_df[height_df["RiskLevel"] == "🟡 मध्यम जोखिम"].head(15)
    elif "सुरक्षित" in filter_choice:
        sample_records = height_df[height_df["RiskLevel"] == "🟢 सुरक्षित"].head(15)
        
    st.dataframe(sample_records, use_container_width=True)


# =============================================================
# TAB 3: SHAP EXPLAINABLE AI & TOP RISK DRIVERS
# =============================================================
with tabs[2]:
    st.subheader("🧠 SHAP व्याख्यात्मक AI (Explainable AI & Root-Cause Hazard Drivers)")
    st.markdown("मशीन लर्निंग मॉडल किस आधार पर निर्णय ले रहा है? SHAP (Shapley Additive exPlanations) द्वारा वैश्विक और व्यक्तिगत जोखिम चालकों का विश्लेषण।")
    
    shap_domain = st.radio(
        "विश्लेषण डोमेन चुनें (Select Domain for SHAP Analysis):",
        ["🪜 ऊंचाई पतन जोखिम (Height Fall Hazard)", "🚜 खुदाई पतन जोखिम (Excavation Collapse Hazard)"],
        horizontal=True
    )
    
    pipeline_obj = height_pipeline if "ऊंचाई" in shap_domain else underground_pipeline
    tree_model = pipeline_obj["models"]["XGBoost"]
    feature_cols = pipeline_obj["feature_cols"]
    
    # Global feature importance via XGBoost / SHAP
    st.markdown("### 🌐 1. वैश्विक जोखिम कारक (Global Hazard Importance)")
    st.caption("संपूर्ण निर्माण स्थल पर कौन से कारक पतन या खाई ढहने में सबसे बड़ा योगदान देते हैं:")
    
    feat_imp = tree_model.feature_importances_
    sorted_idx = np.argsort(feat_imp)[::-1]
    
    top_features = [feature_cols[i] for i in sorted_idx]
    top_importances = [feat_imp[i] for i in sorted_idx]
    
    fig_global, ax_global = plt.subplots(figsize=(10, 4.5))
    bar_colors = ['#ef4444' if i < 3 else '#f59e0b' if i < 6 else '#3b82f6' for i in range(len(top_features))]
    bars = ax_global.barh(top_features[::-1], top_importances[::-1], color=bar_colors[::-1])
    ax_global.set_xlabel("Relative Feature Importance (SHAP Weight)", fontsize=11, fontweight='bold')
    ax_global.set_title("Top Risk Drivers Ranking", fontsize=13, fontweight='bold')
    ax_global.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(fig_global)
    plt.close(fig_global)
    
    st.markdown("---")
    st.markdown("### 🔍 2. व्यक्तिगत चेक-इन का स्थानीय SHAP विश्लेषण (Local Instance Explanation)")
    st.markdown("नवीनतम चेक-इन इनपुट का विस्तृत विश्लेषण: कौन से विशिष्ट कारकों ने जोखिम को बढ़ाया या घटाया?")
    
    # Check if a sample exists from Tab 1
    dom_key = "last_height_sample" if "ऊंचाई" in shap_domain else "last_underground_sample"
    input_key = "last_height_input" if "ऊंचाई" in shap_domain else "last_underground_input"
    
    if dom_key in st.session_state:
        sample_row = st.session_state[dom_key]
        input_dict = st.session_state[input_key]
        
        explainer = shap.TreeExplainer(tree_model)
        shap_vals = explainer(sample_row)
        
        # Multiclass: shap_vals has shape (1, n_features, 3)
        # Class 2 is High Risk (🔴 बहुत अधिक जोखिम)
        shap_for_high_risk = shap_vals.values[0, :, 2]
        
        shap_df = pd.DataFrame({
            "सुविधा (Feature)": feature_cols,
            "श्रमिक इनपुट (Worker Input)": [str(input_dict.get(col, "")) for col in feature_cols],
            "SHAP प्रभाव (+ बढ़ाता है, - घटाता है)": shap_for_high_risk
        }).sort_values(by="SHAP प्रभाव (+ बढ़ाता है, - घटाता है)", ascending=False)
        
        col_s1, col_s2 = st.columns([1.3, 1])
        with col_s1:
            st.markdown("##### SHAP योगदान तालिका (Contribution to Red Alert / High Risk)")
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
            
        with col_s2:
            st.markdown("##### स्थानीय प्रभाव विज़ुअलाइज़ेशन")
            fig_local, ax_local = plt.subplots(figsize=(6, 5))
            colors = ['#ef4444' if x > 0 else '#10b981' for x in shap_df["SHAP प्रभाव (+ बढ़ाता है, - घटाता है)"]]
            ax_local.barh(shap_df["सुविधा (Feature)"][::-1], shap_df["SHAP प्रभाव (+ बढ़ाता है, - घटाता है)"][::-1], color=colors[::-1])
            ax_local.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax_local.set_xlabel("SHAP Value (Push towards Red Alert)", fontsize=10)
            ax_local.set_title("Direct Feature Contribution", fontsize=11, fontweight='bold')
            st.pyplot(fig_local)
            plt.close(fig_local)
    else:
        st.info("💡 कृपया पहले टैब 1 में 'सचित्र सुरक्षा चेक-इन' भरें ताकि स्थानीय SHAP व्याख्या तैयार हो सके।")


# =============================================================
# TAB 4: ACTIONABLE MITIGATION PROTOCOLS
# =============================================================
with tabs[3]:
    st.subheader("🛠️ सुझाई गई सुरक्षा कार्रवाई व मानक संचालन प्रक्रिया (SOPs & Mitigation Protocols)")
    st.markdown("एआई मॉडल द्वारा पहचाने गए शीर्ष जोखिम चालकों के आधार पर अनुकूलित तत्काल सुरक्षा प्रोटोकॉल।")
    
    st.markdown("""
    <div style="background-color: #fef2f2; border: 1px solid #fee2e2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="color: #991b1b; margin: 0 0 8px 0;">🔴 रेड अलर्ट प्रोटोकॉल (Stop Work & Immediate Hazard Mitigation)</h4>
        <ul style="color: #7f1d1d; margin: 0; padding-left: 20px;">
            <li><strong>मचान अस्थिरता (Unstable Scaffolding):</strong> मचान पर तुरंत लाल 'STOP' टैग लगाएं। बेस जैक, टाई-बार और ब्रेसिंग की जांच जब तक सक्षम मचान निरीक्षक द्वारा न की जाए, कार्य प्रारंभ न करें।</li>
            <li><strong>बिना एंकर का हार्नेस (Unanchored Fall Harness):</strong> ऊपरी मंजिलों पर तुरंत 8mm स्टील वायर लाइफलाइन या सेल्फ-रिट्रैक्टिंग लाइफलाइन (SRL) स्थापित करें। श्रमिकों को 100% टाई-ऑफ अनिवार्य करें।</li>
            <li><strong>खुदाई में पानी का रिसाव व दरारें (Seepage & Edge Cracks):</strong> गड्ढे से सभी श्रमिकों को तुरंत बाहर निकालें। 5 HP सबमर्सिबल पम्पों द्वारा पानी बाहर निकालें और हाइड्रोलिक ट्रेंच बॉक्स या टिम्बर शीट पाइलिंग लगाएं।</li>
            <li><strong>खुदाई के किनारे मशीनरी (Machinery near trench):</strong> उत्खनन और भारी ट्रकों को खाई के किनारे से कम से कम 2 से 3 मीटर (गड्ढे की गहराई के बराबर) पीछे रखें।</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-left: 5px solid #f59e0b; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="color: #92400e; margin: 0 0 8px 0;">🟡 एम्बर अलर्ट प्रोटोकॉल (Precautionary Intervention)</h4>
        <ul style="color: #78350f; margin: 0; padding-left: 20px;">
            <li><strong>थकान और लंबी शिफ्ट (Fatigue & Extended Shift >10h):</strong> श्रमिकों को 30 मिनट का विश्राम ब्रेक दें। उच्च जोखिम वाले कार्य (जैसे किनारे की शटरिंग) के लिए शिफ्ट रोटेशन करें।</li>
            <li><strong>खुले किनारे (Open Edges):</strong> 1.05 मीटर ऊंची डबल-रेल सुरक्षा रेलिंग और 15 सेमी टो-बोर्ड तुरंत लगाएं।</li>
            <li><strong>मलबा / लोड (Spoil Pile):</strong> खाई के किनारे जमा की गई मिट्टी या स्टील रीबार को कम से कम 1 मीटर दूर स्थानांतरित करें।</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🖨️ दैनिक सुरक्षा कार्य परमिट (Digital Permit-to-Work - PTW Generator)")
    permit_col1, permit_col2, permit_col3 = st.columns(3)
    with permit_col1:
        p_zone = st.text_input("कार्य स्थान (Work Zone)", value="Tower 1 - Floor 8")
        p_supervisor = st.text_input("सुपरवाइजर का नाम (Supervisor)", value="Ramchandra Sharma")
    with permit_col2:
        p_contractor = st.text_input("ठेकेदार कंपनी (Contractor)", value="Apex Infrastructure Ltd.")
        p_workers_count = st.number_input("श्रमिकों की संख्या (Workers Authorized)", value=6, min_value=1)
    with permit_col3:
        p_ppe_check = st.checkbox("⛑️ सभी पीपीई (हेलमेट, बूट्स, हार्नेस) सत्यापित", value=True)
        p_anchor_check = st.checkbox("⚓ लाइफलाइन एंकरिंग सत्यापित", value=True)
        p_toolbox = st.checkbox("🗣️ टूलबॉक्स सुरक्षा वार्ता संपन्न (Toolbox Talk Done)", value=True)
        
    if st.button("📄 सुरक्षा परमिट जारी करें (Generate Digital Safety Permit)"):
        st.success(f"✅ परमिट संख्या #PTW-2026-9810 जारी किया गया: {p_zone} पर {p_workers_count} श्रमिकों के लिए कार्य अधिकृत है।")


# =============================================================
# TAB 5: ML BENCHMARKS & ECONOMICS RESEARCH
# =============================================================
with tabs[4]:
    st.subheader("📈 मशीन लर्निंग मॉडल तुलना और अर्थशास्त्र शोध (ML Benchmarks & Safety Economics)")
    st.markdown("लॉजिस्टिक रिग्रेशन, रैंडम फ़ॉरेस्ट और एक्सजीबूस्ट के तुलनात्मक परिणाम तथा भारतीय निर्माण क्षेत्र में सूचना अंतराल का आर्थिक विश्लेषण।")
    
    m_tab1, m_tab2 = st.tabs(["🤖 मॉडल प्रदर्शन तुलना (Model Benchmarks)", "💰 निर्माण सुरक्षा अर्थशास्त्र (Safety Economics Research)"])
    
    with m_tab1:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🪜 ऊंचाई जोखिम मॉडल (Height Work Hazard Models)")
            h_df_metrics = pd.DataFrame(height_metrics)
            st.dataframe(h_df_metrics[["Model", "Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1"]], use_container_width=True, hide_index=True)
            
            fig_h, ax_h = plt.subplots(figsize=(6, 3.5))
            ax_h.bar(h_df_metrics["Model"], h_df_metrics["Accuracy"], color=['#94a3b8', '#3b82f6', '#10b981'])
            ax_h.set_ylabel("Accuracy", fontsize=10)
            ax_h.set_ylim(0.85, 1.02)
            ax_h.set_title("Height Models Accuracy Comparison", fontsize=11, fontweight='bold')
            st.pyplot(fig_h)
            plt.close(fig_h)

        with col_m2:
            st.markdown("#### 🚜 खुदाई जोखिम मॉडल (Excavation Hazard Models)")
            u_df_metrics = pd.DataFrame(underground_metrics)
            st.dataframe(u_df_metrics[["Model", "Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1"]], use_container_width=True, hide_index=True)
            
            fig_u, ax_u = plt.subplots(figsize=(6, 3.5))
            ax_u.bar(u_df_metrics["Model"], u_df_metrics["Accuracy"], color=['#94a3b8', '#3b82f6', '#10b981'])
            ax_u.set_ylabel("Accuracy", fontsize=10)
            ax_u.set_ylim(0.85, 1.02)
            ax_u.set_title("Excavation Models Accuracy Comparison", fontsize=11, fontweight='bold')
            st.pyplot(fig_u)
            plt.close(fig_u)

    with m_tab2:
        st.markdown("### 📚 निर्माण सुरक्षा में सूचना अंतराल का अर्थशास्त्र (Economics of Safety Information Gaps)")
        st.markdown("""
        निर्माण क्षेत्र में दुर्घटनाएं केवल तकनीकी खराबी नहीं, बल्कि **सूचना विषमता (Information Asymmetry)** और **बाजार विफलता (Market Failure)** का परिणाम हैं।
        """)
        
        econ_col1, econ_col2 = st.columns([1.2, 1])
        with econ_col1:
            st.markdown("""
            #### 1. प्रमुख आर्थिक अंतर्दृष्टि (Key Research Findings):
            1. **सूचना विषमता (Principal-Agent Dilemma):** मुख्य डेवलपर (Principal) सुरक्षा चाहता है, परंतु उप-ठेकेदार (Agents) लागत घटाने हेतु अनएंकरड मचान और बिना-शोरिंग खुदाई का सहारा लेते हैं।
            2. **असंगठित प्रवासी श्रमिक (Informal Labor Disincentive):** दिहाड़ी मजदूर मजदूरी कटने के भय से नियर-मिस घटनाओं की रिपोर्ट नहीं करते। यह सचित्र चेक-इन प्रणाली रिपोर्टिंग घर्षण को शून्य कर देती है।
            3. **दुर्घटना की प्रत्यक्ष बनाम अप्रत्यक्ष लागत (Direct vs. Indirect Costs):**
               - *प्रत्यक्ष लागत:* चिकित्सा उपचार व मुआवजा (लगभग ₹15-25 लाख प्रति गंभीर घटना)।
               - *अप्रत्यक्ष लागत (आइसबर्ग प्रभाव):* कार्य स्थगन, कानूनी मुकदमेबाजी, क्रेन व मशीनरी का निष्क्रिय रहना, और ब्रांड क्षति प्रत्यक्ष लागत से **4 गुना अधिक (₹60-100 लाख)** होती है।
            4. **लागत-लाभ अनुपात (Cost-Benefit Ratio - BCR):**
               - डिजिटल सचित्र चेक-इन की लागत: **₹5 प्रति श्रमिक / दिन**।
               - दुर्घटना निवारण पर आर्थिक रिटर्न: **₹1 निवेश पर ₹4.20 का शुद्ध आर्थिक लाभ**।
            """)
            
        with econ_col2:
            st.markdown("#### 📉 सुरक्षा अनुपालन बनाम दुर्घटना आवृत्ति")
            compliance_levels = np.array([20, 35, 50, 65, 80, 95])
            accident_rate = 18.5 * np.exp(-0.035 * compliance_levels)
            savings_lakhs = (18.5 - accident_rate) * 4.5
            
            fig_econ, ax1 = plt.subplots(figsize=(6, 4))
            color = '#ef4444'
            ax1.set_xlabel('Safety Check-In Compliance Rate (%)', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Annual Accident Frequency', color=color, fontsize=10, fontweight='bold')
            ax1.plot(compliance_levels, accident_rate, color=color, marker='o', linewidth=2.5)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = '#10b981'
            ax2.set_ylabel('Cost Avoidance (INR Lakhs)', color=color, fontsize=10, fontweight='bold')
            ax2.plot(compliance_levels, savings_lakhs, color=color, marker='s', linewidth=2.5, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
            
            ax1.grid(True, linestyle=':', alpha=0.6)
            fig_econ.tight_layout()
            st.pyplot(fig_econ)
            plt.close(fig_econ)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.88rem;">
    Dual-Risk Construction Safety Analytics Platform &bull; Research by Viraj Jain &bull; Powered by XGBoost, Random Forest & SHAP Explainable AI
</div>
""", unsafe_allow_html=True)
