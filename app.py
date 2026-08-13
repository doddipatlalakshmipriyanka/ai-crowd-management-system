import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CROWDGURD AI-Intelligent Crowd Management System",
    page_icon="👥",
    layout="wide"
)
# =========================================================
# BACKGROUND + WHITE CONTENT
# =========================================================

st.markdown(
    """
    <style>

    /* Full application background */
    .stApp {
        background-image:
            linear-gradient(
                rgba(0, 0, 20, 0.55),
                rgba(0, 0, 20, 0.55)
            ),
            url("https://ai-crowd-management-system-hfciltzzvgjywdtacvjbah.streamlit.app/");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main content text */
    .main .block-container {
        color: white !important;
    }

    /* All normal text */
    .main .block-container p,
    .main .block-container li,
    .main .block-container span,
    .main .block-container label {
        color: white !important;
    }

    /* Headings */
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6 {
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(3, 12, 35, 0.95);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Information boxes */
    div[data-testid="stAlert"] {
        color: white !important;
    }

    div[data-testid="stAlert"] p {
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        color: white !important;
        background-color: #0878d1 !important;
        border: 1px solid #4db8ff !important;
        border-radius: 10px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #005fa8 !important;
        border-color: white !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: rgba(0, 20, 50, 0.70);
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# =========================================================
# LOGO
# =========================================================

import os

logo_path = os.path.join("image", "logo.jpeg")

if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([3, 2, 3])

    with col2:
        st.image(
            logo_path,
            width=600
        )
else:
    st.warning("Logo image not found.")
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 Artificial Intelligence Careers for Women (AICW)")
    

    st.title(" 🎓 Capstone Project")

    st.markdown("---")

    st.markdown("### 👩‍💻 Team Members")

    st.write(
        "1. D.L. Priyanka  \n"
        "doddipatlapriyanka1@gmail.com"
    )

    st.write(
        "2. R.S.V. Madhulika  \n"
        "rayudusrividyamadhulika@gmail.com"
    )

    st.write(
        "3. Y. Leela Devi  \n"
        "leelayejarla@gmail.com"
    )

    st.write(
        "4. K. Vyjayanthi  \n"
        "vyshu366@gmail.com"
    )

    st.markdown("---")

    st.markdown("### 🏫 College Name")

    st.write(
        "VSM COLLEGE OF ENGINEERING"
    )

    st.write(
        "Ramachandrapuram"
    )

    st.markdown("---")

    st.markdown("### 👨‍🏫 Guide Name")

    st.write("Abdul Aziz Md, Lead - AICW (South)")


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        👥 CROWDGURD AI-Intelligent Crowd Management System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="text-align:center;">
        AI-Powered Crowd Detection, Counting and Risk Analysis
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# PROJECT TITLE
# =========================================================

st.header("🚨 CROWDGURD AI-Intelligent Crowd Management System")

st.write(
    """
    An intelligent computer vision-based application designed to
    monitor crowds, detect people, count individuals and identify
    different levels of crowd risk.
    """
)


# =========================================================
# OPEN PROJECT APPLICATION
# =========================================================

st.subheader("🚀 Project Application")

st.write(
    """
    The complete AI Crowd Management application is available
    on the next page. Click the button below to continue.
    """
)

# IMPORTANT:
# This opens the second page in the SAME localhost application.
# It does NOT use the deployed Streamlit URL.

if st.button(
    "🛡️ OPEN CROWDGUARD AI APPLICATION",
    type="primary",
    use_container_width=True
):
    st.switch_page(
        "pages/1_Crowd_Management_Application.py"
    )


st.markdown("---")


# =========================================================
# PROJECT DESCRIPTION
# =========================================================

st.header("📋 Project Description")

st.write(
    """
    The CrowdGurd AI is an intelligent Crowd Management System uses Artificial Intelligence
    and Computer Vision to analyze crowd situations.

    The system uses the YOLO object detection model to detect
    people from images, videos and live camera input.

    It automatically counts the detected people and classifies
    the crowd into different risk levels:

    • LOW  
    • MEDIUM  
    • HIGH

    The system is designed to support crowd monitoring and
    early identification of potentially crowded situations.
    """
)


# =========================================================
# KEY FEATURES
# =========================================================

st.header("✨ Key Features")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        ### 📷 Image Analysis

        • Upload crowd images  
        • Detect people using AI  
        • Count detected people  
        • Display crowd risk level
        """
    )

with col2:

    st.markdown(
        """
        ### 🎥 Video & Live Camera

        • Analyze crowd videos  
        • Live camera monitoring  
        • Real-time people detection  
        • People counting
        """
    )


col3, col4 = st.columns(2)

with col3:

    st.markdown(
        """
        ### 🚨 Risk Detection

        • LOW crowd level  
        • MEDIUM crowd level  
        • HIGH crowd level  
        • High-crowd alert
        """
    )

with col4:

    st.markdown(
        """
        ### 📍 Location Monitoring

        • GPS location support  
        • Crowd monitoring location  
        • Location information  
        • Alert location information
        """
    )


st.markdown("---")


# =========================================================
# PROJECT GOAL
# =========================================================

st.header("🎯 Project Goal")

st.info(
    """
    To provide an AI-powered solution for monitoring crowds,
    identifying potentially high-risk crowd situations and
    supporting safer public spaces.
    """
)


# =========================================================
# TECHNOLOGIES
# =========================================================

st.header("🛠️ Technologies Used")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.info("🐍 Python")

with tech2:
    st.info("🤖 YOLO")

with tech3:
    st.info("👁️ Computer Vision")

with tech4:
    st.info("🌐 Streamlit")


st.markdown("---")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <h4 style="text-align:center;">
        Artificial Intelligence Careers for Women (AICW)
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Capstone Project • VSM College of Engineering
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Guided by Abdul Aziz Md, Lead - AICW (South)
    </p>
    """,
    unsafe_allow_html=True
)
