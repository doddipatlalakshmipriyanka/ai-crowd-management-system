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
# BACKGROUND IMAGE
# =========================================================

import os

background_path = os.path.join(
    "image",
    "background.jpeg"
)

if os.path.exists(background_path):

    with open(background_path, "rb") as f:
        background_data = f.read()

    import base64

    background_base64 = base64.b64encode(
        background_data
    ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(5, 15, 35, 0.82),
                    rgba(5, 15, 35, 0.82)
                ),
                url("data:image/jpeg;base64,{background_base64}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

else:
    st.warning("Background image not found.")
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
