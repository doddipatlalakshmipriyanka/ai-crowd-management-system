import os
import time
import uuid
import threading
import tempfile
import smtplib
from pathlib import Path
from email.message import EmailMessage

import av
import cv2
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from aiortc.contrib.media import MediaRecorder
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CrowdGuard AI-Intelligent Crowd Management System",
    page_icon="👥",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1000px;
            margin: auto;
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Crowd Threshold Settings")

crowd_threshold = st.sidebar.number_input(
    "🚨 High Crowd Threshold",
    min_value=1,
    max_value=500,
    value=20,
    step=1,
    help="Set the number of people above which the crowd is considered HIGH.",
)

st.sidebar.info(
    f"🚨 HIGH crowd when people ≥ {crowd_threshold}"
)

CAMERA_CONFIDENCE = 0.25
CAMERA_IMAGE_SIZE = 416

HIGH_THRESHOLD = crowd_threshold
HIGH_FRAMES_REQUIRED = 10
ALERT_COOLDOWN_SECONDS = 300

CAMERA_DISPLAY_WIDTH = 650

RECORD_DIR = Path("camera_records")
RECORD_DIR.mkdir(exist_ok=True)

IMAGE_DIR = Path("image")


# ============================================================
# SESSION STATE
# ============================================================

if "frame_store" not in st.session_state:
    st.session_state.frame_store = {
        "frame": None,
        "lock": threading.Lock(),
    }

if "camera_store" not in st.session_state:
    st.session_state.camera_store = {
        "people": 0,
        "risk": "LOW",
        "high_frames": 0,
        "last_alert": 0.0,
        "lock": threading.Lock(),
    }

if "gps_store" not in st.session_state:
    st.session_state.gps_store = {
        "latitude": None,
        "longitude": None,
        "accuracy": None,
        "location_text": None,
    }

if "record_store" not in st.session_state:
    st.session_state.record_store = {
        "record_id": str(uuid.uuid4()),
    }

if "captured_photo" not in st.session_state:
    st.session_state.captured_photo = None

if "captured_photo_people" not in st.session_state:
    st.session_state.captured_photo_people = 0

if "captured_photo_risk" not in st.session_state:
    st.session_state.captured_photo_risk = "LOW"

# ------------------------------------------------------------
# NEW: prevent repeated alerts for same uploaded/captured item
# ------------------------------------------------------------

if "live_photo_alert_sent" not in st.session_state:
    st.session_state.live_photo_alert_sent = False

if "uploaded_image_alert_id" not in st.session_state:
    st.session_state.uploaded_image_alert_id = None

if "uploaded_video_alert_id" not in st.session_state:
    st.session_state.uploaded_video_alert_id = None


frame_store = st.session_state.frame_store
camera_store = st.session_state.camera_store
gps_store = st.session_state.gps_store

record_id = st.session_state.record_store["record_id"]

record_file = RECORD_DIR / f"{record_id}_crowd_video.flv"


# ============================================================
# TITLE
# ============================================================

st.title("👥 CrowdGuard AI-Intelligent Crowd Management System")

st.write(
    "AI-powered crowd detection, people counting, risk analysis, "
    "live photo/video capture and location-based HIGH-crowd alerts."
)


# ============================================================
# BANNER
# ============================================================

st.markdown("---")
st.subheader("🏠 Welcome")

banner_candidates = [
    IMAGE_DIR / "banner (3).jpg",
    IMAGE_DIR / "banner (2).jpg",
    IMAGE_DIR / "banner.jpg",
]

banner_path = None

for path in banner_candidates:
    if path.exists():
        banner_path = path
        break

if banner_path:
    st.image(
        str(banner_path),
        use_container_width=True,
    )
else:
    st.warning(
        "⚠️ Banner image not found. "
        "Put your banner image inside the image folder."
    )

st.write(
    "This system detects people using YOLO and classifies crowd "
    "risk as LOW, MEDIUM or HIGH."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.info(
        "📷 **Live Photo**\n\n"
        "Capture the current processed camera frame."
    )

with c2:
    st.info(
        "🎥 **Live Video**\n\n"
        "Record the live camera stream."
    )

with c3:
    st.info(
        "🚨 **HIGH Alert**\n\n"
        "Send crowd count and location by email."
    )


# ============================================================
# YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = Path("yolov8s.pt")

    if not model_path.exists():

        raise FileNotFoundError(
            "yolov8s.pt was not found. "
            "Put yolov8s.pt in the same folder as app.py."
        )

    return YOLO(str(model_path))


try:

    model = load_model()

except Exception as e:

    st.error("❌ YOLO model could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# RISK FUNCTION
# ============================================================

def get_risk(count, threshold):

    medium_threshold = max(
        1,
        threshold // 2
    )

    if count < medium_threshold:

        return "LOW"

    elif count < threshold:

        return "MEDIUM"

    else:

        return "HIGH"


# ============================================================
# EMAIL CONFIGURATION
# ============================================================

def load_email_config():

    try:

        secrets = st.secrets

        return {
            "sender_email": secrets.get(
                "ALERT_EMAIL",
                ""
            ),

            "sender_password": secrets.get(
                "ALERT_PASSWORD",
                ""
            ),

            "admin_email": secrets.get(
                "ADMIN_EMAIL",
                ""
            ),
        }

    except Exception:

        return {
            "sender_email": "",
            "sender_password": "",
            "admin_email": "",
        }


EMAIL_CONFIG = load_email_config()


# ============================================================
# EMAIL ALERT
# ============================================================

def send_crowd_alert(
    people_count,
    location_text=None,
    latitude=None,
    longitude=None,
    source="Live Camera",
):

    try:

        sender_email = EMAIL_CONFIG["sender_email"]
        sender_password = EMAIL_CONFIG["sender_password"]
        admin_email = EMAIL_CONFIG["admin_email"]

        if not sender_email:

            print("ERROR: ALERT_EMAIL is missing.")
            return False

        if not sender_password:

            print("ERROR: ALERT_PASSWORD is missing.")
            return False

        if not admin_email:

            print("ERROR: ADMIN_EMAIL is missing.")
            return False

        if location_text:

            final_location = location_text

        elif latitude is not None and longitude is not None:

            final_location = (
                f"Latitude: {float(latitude):.6f}\n"
                f"Longitude: {float(longitude):.6f}"
            )

        else:

            final_location = "GPS location unavailable."

        message = EmailMessage()

        message["Subject"] = (
            "🚨 HIGH CROWD ALERT - AI Crowd Management System"
        )

        message["From"] = sender_email
        message["To"] = admin_email

        message.set_content(
            f"""HIGH CROWD ALERT
==============================

Source:
{source}

Detected People:
{people_count}

Risk Level:
HIGH

Location:
{final_location}

Please verify the situation and take appropriate action.

AI Crowd Management System
"""
        )

        print("Connecting to Gmail SMTP...")

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=20
        ) as server:

            print("Logging into Gmail...")

            server.login(
                sender_email,
                sender_password
            )

            print("Sending email...")

            server.send_message(message)

        print("EMAIL SENT SUCCESSFULLY.")

        return True

    except Exception as e:

        print("EMAIL ALERT ERROR:")
        print(type(e).__name__)
        print(str(e))

        return False


# ============================================================
# CENTRAL HIGH-CROWD ALERT FUNCTION
# ============================================================

def send_high_crowd_alert_once(
    people_count,
    source,
    alert_id=None,
):

    location_text = gps_store.get(
        "location_text"
    )

    latitude = gps_store.get(
        "latitude"
    )

    longitude = gps_store.get(
        "longitude"
    )

    # --------------------------------------------------------
    # LIVE PHOTO
    # --------------------------------------------------------

    if source == "Live Photo Capture":

        if st.session_state.live_photo_alert_sent:

            return False

        alert_sent = send_crowd_alert(
            people_count=people_count,
            location_text=location_text,
            latitude=latitude,
            longitude=longitude,
            source=source,
        )

        if alert_sent:

            st.session_state.live_photo_alert_sent = True

        return alert_sent

    # --------------------------------------------------------
    # UPLOADED IMAGE
    # --------------------------------------------------------

    if source == "Uploaded Image":

        if (
            alert_id is not None
            and st.session_state.uploaded_image_alert_id == alert_id
        ):

            return False

        alert_sent = send_crowd_alert(
            people_count=people_count,
            location_text=location_text,
            latitude=latitude,
            longitude=longitude,
            source=source,
        )

        if alert_sent:

            st.session_state.uploaded_image_alert_id = alert_id

        return alert_sent

    # --------------------------------------------------------
    # UPLOADED VIDEO
    # --------------------------------------------------------

    if source == "Uploaded Video":

        if (
            alert_id is not None
            and st.session_state.uploaded_video_alert_id == alert_id
        ):

            return False

        alert_sent = send_crowd_alert(
            people_count=people_count,
            location_text=location_text,
            latitude=latitude,
            longitude=longitude,
            source=source,
        )

        if alert_sent:

            st.session_state.uploaded_video_alert_id = alert_id

        return alert_sent

    # --------------------------------------------------------
    # OTHER SOURCES
    # --------------------------------------------------------

    return send_crowd_alert(
        people_count=people_count,
        location_text=location_text,
        latitude=latitude,
        longitude=longitude,
        source=source,
    )


def reverse_geocode(latitude, longitude):

    try:
        url = "https://nominatim.openstreetmap.org/reverse"

        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "zoom": 18,
            "addressdetails": 1,
            "accept-language": "en"
        }

        headers = {
            "User-Agent": "CrowdGurdAI/1.0"
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return "Location name could not be determined"

        data = response.json()

        # First try the complete readable address
        location = data.get("display_name")

        if location:
            return location

        # Otherwise construct a readable location
        address = data.get("address", {})

        parts = []

        for key in [
            "amenity",
            "building",
            "road",
            "village",
            "town",
            "city",
            "state"
        ]:
            value = address.get(key)

            if value and value not in parts:
                parts.append(value)

        if parts:
            return ", ".join(parts)

        return "Location name could not be determined"

    except Exception as e:

        print("Reverse geocoding error:", e)

        return "Location name could not be determined"
# ============================================================
# CAMERA FRAME CALLBACK
# ============================================================

def camera_frame_callback(
    frame: av.VideoFrame
):

    image = frame.to_ndarray(
        format="bgr24"
    )

    try:

        results = model(
            image,
            classes=[0],
            conf=CAMERA_CONFIDENCE,
            imgsz=CAMERA_IMAGE_SIZE,
            verbose=False,
        )

    except Exception as e:

        print(
            "YOLO camera error:",
            e
        )

        return frame

    people_count = 0

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            try:

                class_id = int(
                    box.cls[0]
                )

            except Exception:

                continue

            if class_id != 0:

                continue

            people_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0],
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                "Person",
                (
                    x1,
                    max(y1 - 8, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    risk = get_risk(
        people_count,
        crowd_threshold
    )

    # --------------------------------------------------------
    # Store processed frame
    # --------------------------------------------------------

    with frame_store["lock"]:

        frame_store["frame"] = (
            image.copy()
        )

    # --------------------------------------------------------
    # Camera statistics
    # --------------------------------------------------------

    should_alert = False

    alert_latitude = gps_store.get(
        "latitude"
    )

    alert_longitude = gps_store.get(
        "longitude"
    )

    alert_location = gps_store.get(
        "location_text"
    )

    with camera_store["lock"]:

        camera_store["people"] = (
            people_count
        )

        camera_store["risk"] = risk

        if people_count >= HIGH_THRESHOLD:

            camera_store["high_frames"] += 1

        else:

            camera_store["high_frames"] = 0

        now = time.time()

        if (
            camera_store["high_frames"]
            >= HIGH_FRAMES_REQUIRED
            and
            now - camera_store["last_alert"]
            >= ALERT_COOLDOWN_SECONDS
        ):

            camera_store["last_alert"] = now

            should_alert = True

    # --------------------------------------------------------
    # Send HIGH crowd alert
    # --------------------------------------------------------

    if should_alert:

        threading.Thread(
            target=send_crowd_alert,
            kwargs={
                "people_count": people_count,
                "location_text": alert_location,
                "latitude": alert_latitude,
                "longitude": alert_longitude,
                "source": "Live Camera",
            },
            daemon=True,
        ).start()

    # --------------------------------------------------------
    # Risk color
    # --------------------------------------------------------

    if risk == "LOW":

        color = (
            0,
            255,
            0,
        )

    elif risk == "MEDIUM":

        color = (
            0,
            255,
            255,
        )

    else:

        color = (
            0,
            0,
            255,
        )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    cv2.putText(
        image,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        image,
        f"Risk: {risk}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    if gps_store.get(
        "latitude"
    ) is not None:

        cv2.putText(
            image,
            "GPS: ON",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    else:

        cv2.putText(
            image,
            "GPS: OFF",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if risk == "HIGH":

        cv2.putText(
            image,
            "HIGH CROWD ALERT!",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24",
    )


# ============================================================
# WEBRTC RECORDING
# ============================================================

def out_recorder_factory():

    return MediaRecorder(
        str(record_file),
        format="flv",
    )


# ============================================================
# LIVE CAMERA
# ============================================================

st.markdown("---")

st.subheader("📷🎥 Live Camera")

st.write(
    "Click START to activate your camera. "
    "The camera must be started before taking a live photo "
    "or recording live video."
)

camera_left, camera_center, camera_right = st.columns(
    [1, 2, 1]
)

with camera_center:

    try:

        ctx = webrtc_streamer(
            key="crowd-live-camera-final",

            mode=WebRtcMode.SENDRECV,

            video_frame_callback=camera_frame_callback,

            out_recorder_factory=out_recorder_factory,

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            },
        )

    except Exception as e:

        st.error(
            "❌ Camera could not be started."
        )

        st.code(
            str(e)
        )

        ctx = None


# ============================================================
# CAMERA STATUS
# ============================================================

if ctx is not None:

    if ctx.state.playing:

        st.success(
            "🟢 Camera is LIVE. "
            "YOLO detection and video recording are active."
        )

    else:

        st.info(
            "🔴 Camera is stopped. "
            "Click START above to activate it."
        )


# ============================================================
# LIVE PHOTO CAPTURE
# ============================================================

st.markdown("---")

st.subheader("📸 Live Photo Capture")

st.write(
    "Start the camera first, wait for the live video to appear, "
    "then click **Capture Current Image**."
)

camera_running = (
    ctx is not None
    and ctx.state.playing
)

capture_photo = st.button(
    "📸 Capture Current Image",
    key="capture_live_photo",
    disabled=not camera_running,
)


# ============================================================
# CAPTURE PHOTO
# ============================================================

if capture_photo:

    # New capture = allow a new alert
    st.session_state.live_photo_alert_sent = False

    with frame_store["lock"]:

        if frame_store["frame"] is None:

            current_frame = None

        else:

            current_frame = (
                frame_store["frame"].copy()
            )

    if current_frame is None:

        st.warning(
            "⚠️ Camera is running, but no processed frame "
            "is ready yet. Wait 1–2 seconds and try again."
        )

    else:

        photo_people = camera_store.get(
            "people",
            0
        )

        photo_risk = camera_store.get(
            "risk",
            "LOW"
        )

        st.session_state.captured_photo = (
            current_frame.copy()
        )

        st.session_state.captured_photo_people = (
            photo_people
        )

        st.session_state.captured_photo_risk = (
            photo_risk
        )

        st.success(
            "📸 Photo captured successfully!"
        )

        st.subheader(
            "🖼️ Captured Crowd Image"
        )

        photo_rgb = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            photo_rgb,
            caption="Live Captured Crowd",
            width=700
        )

        photo_col1, photo_col2 = st.columns(2)

        with photo_col1:

            st.metric(
                "👥 People Detected",
                photo_people
            )

        with photo_col2:

            st.metric(
                "⚠️ Crowd Risk",
                photo_risk
            )

        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if photo_risk == "HIGH":

            st.error(
                f"🚨 HIGH CROWD DETECTED! "
                f"People: {photo_people}"
            )

            # ------------------------------------------------
            # SEND EMAIL
            # ------------------------------------------------

            alert_sent = send_high_crowd_alert_once(
                people_count=photo_people,
                source="Live Photo Capture",
            )

            if alert_sent:

                st.success(
                    "📧 HIGH crowd alert email sent successfully."
                )

            elif st.session_state.live_photo_alert_sent:

                st.info(
                    "📧 HIGH crowd alert was already sent "
                    "for this captured photo."
                )

            else:

                st.warning(
                    "⚠️ HIGH crowd detected, "
                    "but the email alert could not be sent."
                )

                st.info(
                    "Please check your Gmail/Streamlit Secrets configuration."
                )

        elif photo_risk == "MEDIUM":

            st.warning(
                f"⚠️ MEDIUM CROWD LEVEL. "
                f"People: {photo_people}"
            )

        else:

            st.success(
                f"✅ Crowd level is LOW. "
                f"People: {photo_people}"
            )


# ============================================================
# SHOW CAPTURED PHOTO
# ============================================================

if st.session_state.captured_photo is not None:

    st.subheader(
        "📸 Captured Live Photo"
    )

    photo_rgb = cv2.cvtColor(
        st.session_state.captured_photo,
        cv2.COLOR_BGR2RGB,
    )

    st.image(
        photo_rgb,
        caption="Captured Crowd Image",
        width=600,
    )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "👥 People Detected",
            st.session_state.captured_photo_people,
        )

    with p2:

        st.metric(
            "⚠️ Crowd Risk",
            st.session_state.captured_photo_risk,
        )


# ============================================================
# DOWNLOAD CAPTURED PHOTO
# ============================================================

if st.session_state.captured_photo is not None:

    try:

        success, encoded_image = cv2.imencode(
            ".jpg",
            st.session_state.captured_photo,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                85,
            ],
        )

        if success:

            photo_bytes = (
                encoded_image.tobytes()
            )

            st.download_button(
                "📥 Download Captured Photo",
                data=photo_bytes,
                file_name="live_crowd_photo.jpg",
                mime="image/jpeg",
                key="download_live_photo",
            )

        else:

            st.error(
                "❌ Could not convert captured photo to JPEG."
            )

    except Exception as e:

        st.error(
            "❌ Error while preparing captured photo."
        )

        st.code(
            str(e)
        )


# ============================================================
# LIVE VIDEO
# ============================================================

st.markdown("---")

st.subheader("🎥 Live Video Capture")

if camera_running:

    st.info(
        "🎥 Live video recording is active. "
        "Keep the camera running and click STOP "
        "when you want to finish recording."
    )

else:

    if (
        record_file.exists()
        and record_file.stat().st_size > 0
    ):

        try:

            video_bytes = (
                record_file.read_bytes()
            )

            st.success(
                "✅ Live crowd video recording completed."
            )

            st.subheader(
                "🎬 Recorded Crowd Video"
            )

            st.video(
                video_bytes
            )

            st.download_button(
                label="📥 Download Live Crowd Video",
                data=video_bytes,
                file_name="live_crowd_video.flv",
                mime="video/x-flv",
                key="download_live_crowd_video",
            )

        except Exception as e:

            st.error(
                "❌ Video file could not be read."
            )

            st.code(
                str(e)
            )

    else:

        st.info(
            "📹 No recorded video yet. "
            "Start the camera, keep it running, "
            "then click STOP."
        )


# ============================================================
# LIVE CAMERA STATUS
# ============================================================

st.markdown("---")

st.subheader("📊 Live Camera Status")

current_people = camera_store["people"]
current_risk = camera_store["risk"]

s1, s2, s3 = st.columns(3)

with s1:

    st.metric(
        "👥 People Detected",
        current_people,
    )

with s2:

    st.metric(
        "⚠️ Current Risk",
        current_risk,
    )

with s3:

    if gps_store["latitude"] is not None:

        st.success(
            "📍 GPS Active"
        )

    else:

        st.warning(
            "📍 GPS Unavailable"
        )


if current_risk == "HIGH":

    st.error(
        "🚨 HIGH CROWD DETECTED!"
    )


# ============================================================
# GPS LOCATION
# ============================================================

st.markdown("---")

st.subheader("📍 GPS Location")

st.write(
    "GPS is independent from the camera. "
    "Click GET GPS LOCATION and allow browser location permission."
)

gps_result = get_geolocation(
    "📍 GET GPS LOCATION"
)


# ============================================================
# PROCESS GPS RESULT
# ============================================================

if isinstance(gps_result, dict):

    if "error" in gps_result:

        error_info = gps_result.get(
            "error"
        )

        if isinstance(error_info, dict):

            error_code = error_info.get(
                "code"
            )

            error_message = error_info.get(
                "message",
                "Unknown GPS error",
            )

            if error_code == 1:

                st.error(
                    "❌ Browser location permission was denied."
                )

            elif error_code == 2:

                st.warning(
                    "⚠️ Location is temporarily unavailable."
                )

            elif error_code == 3:

                st.warning(
                    "⚠️ GPS request timed out. "
                    "Click GET GPS LOCATION again."
                )

            else:

                st.warning(
                    f"⚠️ GPS error: {error_message}"
                )

        else:

            st.warning(
                f"⚠️ GPS error: {error_info}"
            )

    elif gps_result.get("coords"):

        coords = gps_result["coords"]

        lat = coords.get(
            "latitude"
        )

        lon = coords.get(
            "longitude"
        )

        accuracy = coords.get(
            "accuracy"
        )

        if lat is not None and lon is not None:

            lat = float(lat)
            lon = float(lon)

            gps_store["latitude"] = lat
            gps_store["longitude"] = lon

            if accuracy is not None:

                gps_store["accuracy"] = float(
                    accuracy
                )

            location_text = reverse_geocode(
                lat,
                lon,
            )

            if location_text:
                gps_store["location_text"] = location_text
            else:
                gps_store["location_text"] = (
                    "Location name could not be determined"
                )
# ============================================================
# DISPLAY GPS
# ============================================================

latitude = gps_store["latitude"]
longitude = gps_store["longitude"]
accuracy = gps_store["accuracy"]
location_text = gps_store["location_text"]


if latitude is not None and longitude is not None:

    st.success(
        "📍 GPS location detected successfully."
    )

    st.markdown(
        "### 📍 Current Location"
    )

    if location_text:

        st.info(
            f"📍 **{location_text}**"
        )

    else:

        st.info(
            "📍 Location name is being determined..."
        )

    if accuracy is not None:

        st.caption(
            f"GPS accuracy: approximately ±{accuracy:.1f} m"
        )

    gps_map = folium.Map(
        location=[
            latitude,
            longitude,
        ],
        zoom_start=16,
    )

    folium.Marker(
        [
            latitude,
            longitude,
        ],
        popup=location_text or "Current Location",
        tooltip="📍 Crowd Monitoring Location",
    ).add_to(gps_map)

    st_folium(
        gps_map,
        width=700,
        height=350,
        key="current_gps_map",
    )

else:

    st.info(
        "📍 GPS location has not been detected yet. "
        "Click GET GPS LOCATION and allow browser location permission."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("---")

st.subheader("📁 Upload Crowd Image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    key="crowd_image_upload",
)


if uploaded_image is not None:

    # ========================================================
    # UNIQUE IMAGE ID
    # ========================================================

    image_alert_id = (
        f"{uploaded_image.name}_"
        f"{uploaded_image.size}"
    )

    # ========================================================
    # READ IMAGE
    # ========================================================

    image_array = np.frombuffer(
        uploaded_image.getvalue(),
        dtype=np.uint8
    )

    upload_image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if upload_image is None:

        st.error(
            "❌ Unable to read image."
        )

    else:

        # ====================================================
        # IMAGE CROWD DETECTION
        # ====================================================

        results = model.predict(
            source=upload_image,
            classes=[0],
            conf=0.15,
            iou=0.50,
            imgsz=960,
            max_det=1000,
            verbose=False
        )

        people_count = 0

        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        for result in results:

            if result.boxes is None:

                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                if class_id != 0:

                    continue

                people_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(
                    box.conf[0]
                )

                cv2.rectangle(
                    upload_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"Person {confidence:.2f}"
                )

                cv2.putText(
                    upload_image,
                    label,
                    (
                        x1,
                        max(y1 - 8, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        risk = get_risk(
            people_count,
            crowd_threshold
        )

        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        st.subheader(
            "📷 Image Crowd Analysis"
        )

        upload_rgb = cv2.cvtColor(
            upload_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            upload_rgb,
            caption="Detected Crowd",
            width=600
        )

        # ====================================================
        # RESULTS
        # ====================================================

        u1, u2 = st.columns(2)

        with u1:

            st.metric(
                "👥 People Detected",
                people_count
            )

        with u2:

            st.metric(
                "⚠️ Crowd Risk",
                risk
            )

        # ====================================================
        # HIGH / MEDIUM / LOW
        # ====================================================

        if risk == "HIGH":

            st.error(
                f"🚨 HIGH CROWD DETECTED! "
                f"People: {people_count}"
            )

            if location_text:

                st.warning(
                    f"📍 Alert Location: {location_text}"
                )

            else:

                st.warning(
                    "📍 Alert Location: GPS unavailable"
                )

            # ------------------------------------------------
            # SEND EMAIL
            # ------------------------------------------------

            if (
                st.session_state.uploaded_image_alert_id
                != image_alert_id
            ):

                alert_sent = (
                    send_high_crowd_alert_once(
                        people_count=people_count,
                        source="Uploaded Image",
                        alert_id=image_alert_id,
                    )
                )

                if alert_sent:

                    st.success(
                        "📧 HIGH crowd alert email sent successfully."
                    )

                else:

                    st.warning(
                        "⚠️ HIGH crowd detected, "
                        "but the email alert could not be sent."
                    )

                    st.info(
                        "Please check your Gmail/Streamlit Secrets configuration."
                    )

            else:

                st.info(
                    "📧 Alert already sent for this uploaded image."
                )

        elif risk == "MEDIUM":

            st.warning(
                f"⚠️ MEDIUM CROWD LEVEL. "
                f"People: {people_count}"
            )

        else:

            st.success(
                f"✅ LOW CROWD LEVEL. "
                f"People: {people_count}"
            )


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.markdown("---")

st.subheader("🎥 Upload Crowd Video")

uploaded_video = st.file_uploader(
    "Choose a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
    key="crowd_video_upload",
)


if uploaded_video is not None:

    # ========================================================
    # UNIQUE VIDEO ID
    # ========================================================

    video_alert_id = (
        f"{uploaded_video.name}_"
        f"{uploaded_video.size}"
    )

    # ========================================================
    # SAVE UPLOADED VIDEO
    # ========================================================

    file_extension = (
        Path(uploaded_video.name).suffix.lower()
        or ".mp4"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension,
    ) as input_file:

        input_file.write(
            uploaded_video.getvalue()
        )

        input_path = input_file.name

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        st.error(
            "❌ Unable to open video."
        )

        try:

            os.remove(
                input_path
            )

        except OSError:

            pass

    else:

        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:

            fps = 25

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # ====================================================
        # VIDEO AI SETTINGS
        # ====================================================

        VIDEO_CONFIDENCE = 0.15
        VIDEO_IOU = 0.50
        VIDEO_IMAGE_SIZE = 640

        FRAME_SKIP = 2

        MAX_DETECTIONS = 1000

        # ====================================================
        # OUTPUT VIDEO
        # ====================================================

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ).name

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
        )

        # ====================================================
        # DATA
        # ====================================================

        time_data = []
        people_data = []

        max_people = 0

        frame_number = 0

        last_people_count = 0

        # ====================================================
        # PROGRESS
        # ====================================================

        progress = st.progress(
            0
        )

        status = st.empty()

        # ====================================================
        # VIDEO PROCESSING LOOP
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            # =================================================
            # YOLO DETECTION
            # =================================================

            if frame_number % FRAME_SKIP == 0:

                results = model.predict(
                    source=frame,
                    classes=[0],
                    conf=VIDEO_CONFIDENCE,
                    iou=VIDEO_IOU,
                    imgsz=VIDEO_IMAGE_SIZE,
                    max_det=MAX_DETECTIONS,
                    verbose=False,
                )

                people_count = 0

                # =============================================
                # DRAW PERSON DETECTIONS
                # =============================================

                for result in results:

                    if result.boxes is None:

                        continue

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        if class_id != 0:

                            continue

                        people_count += 1

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            f"Person {confidence:.2f}",
                            (
                                x1,
                                max(y1 - 8, 20)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                last_people_count = (
                    people_count
                )

            else:

                people_count = (
                    last_people_count
                )

            # =================================================
            # MAXIMUM PEOPLE
            # =================================================

            max_people = max(
                max_people,
                people_count,
            )

            # =================================================
            # RISK
            # =================================================

            risk = get_risk(
                people_count,
                crowd_threshold,
            )

            # =================================================
            # RISK COLOR
            # =================================================

            if risk == "LOW":

                color = (
                    0,
                    255,
                    0,
                )

            elif risk == "MEDIUM":

                color = (
                    0,
                    255,
                    255,
                )

            else:

                color = (
                    0,
                    0,
                    255,
                )

            # =================================================
            # PEOPLE COUNT
            # =================================================

            cv2.putText(
                frame,
                f"People: {people_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # =================================================
            # RISK
            # =================================================

            cv2.putText(
                frame,
                f"Risk: {risk}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # =================================================
            # HIGH CROWD ALERT ON VIDEO
            # =================================================

            if risk == "HIGH":

                cv2.putText(
                    frame,
                    "HIGH CROWD ALERT!",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # =================================================
            # WRITE OUTPUT FRAME
            # =================================================

            out.write(
                frame
            )

            # =================================================
            # SAVE DATA
            # =================================================

            video_time = (
                frame_number / fps
            )

            time_data.append(
                round(
                    video_time,
                    2,
                )
            )

            people_data.append(
                people_count
            )

            frame_number += 1

            # =================================================
            # PROGRESS
            # =================================================

            if total_frames > 0:

                percentage = min(
                    frame_number / total_frames,
                    1.0,
                )

                progress.progress(
                    percentage
                )

                status.write(
                    f"Processing video... "
                    f"{int(percentage * 100)}% | "
                    f"People: {people_count} | "
                    f"Risk: {risk}"
                )

        # ====================================================
        # CLOSE VIDEO
        # ====================================================

        cap.release()

        out.release()

        progress.empty()

        status.empty()

        st.success(
            "🎉 Video processing completed!"
        )

        # ====================================================
        # PROCESSED VIDEO
        # ====================================================

        st.subheader(
            "🎬 Processed Video"
        )

        with open(
            output_path,
            "rb",
        ) as video_file:

            processed_video = (
                video_file.read()
            )

        st.video(
            processed_video
        )

        # ====================================================
        # FINAL RISK
        # ====================================================

        final_risk = get_risk(
            max_people,
            crowd_threshold,
        )

        # ====================================================
        # FINAL RESULTS
        # ====================================================

        st.subheader(
            "📊 Video Crowd Analysis"
        )

        v1, v2 = st.columns(2)

        with v1:

            st.metric(
                "👥 Maximum People Detected",
                max_people,
            )

        with v2:

            st.metric(
                "⚠️ Maximum Crowd Risk",
                final_risk,
            )

        # ====================================================
        # FINAL ALERT
        # ====================================================

        if final_risk == "HIGH":

            st.error(
                f"🚨 HIGH CROWD DETECTED! "
                f"Maximum people: {max_people}"
            )

            if location_text:

                st.warning(
                    f"📍 Alert Location: "
                    f"{location_text}"
                )

            else:

                st.warning(
                    "📍 Alert Location: GPS unavailable"
                )

            # ------------------------------------------------
            # SEND VIDEO ALERT
            # ------------------------------------------------

            if (
                st.session_state.uploaded_video_alert_id
                != video_alert_id
            ):

                alert_sent = (
                    send_high_crowd_alert_once(
                        people_count=max_people,
                        source="Uploaded Video",
                        alert_id=video_alert_id,
                    )
                )

                if alert_sent:

                    st.success(
                        "📧 HIGH crowd alert email sent successfully."
                    )

                else:

                    st.warning(
                        "⚠️ HIGH crowd detected, "
                        "but the email alert could not be sent."
                    )

                    st.info(
                        "Please check your Gmail/Streamlit Secrets configuration."
                    )

            else:

                st.info(
                    "📧 Alert already sent for this uploaded video."
                )

        elif final_risk == "MEDIUM":

            st.warning(
                f"⚠️ MEDIUM CROWD LEVEL. "
                f"Maximum people: {max_people}"
            )

        else:

            st.success(
                f"✅ Crowd level is LOW. "
                f"Maximum people: {max_people}"
            )

        # ====================================================
        # CROWD GRAPH
        # ====================================================

        if time_data:

            graph_df = pd.DataFrame(
                {
                    "Time (seconds)": time_data,
                    "People": people_data,
                }
            )

            st.subheader(
                "📈 Crowd Count Over Time"
            )

            st.line_chart(
                graph_df,
                x="Time (seconds)",
                y="People",
            )

        # ====================================================
        # DOWNLOAD PROCESSED VIDEO
        # ====================================================

        st.download_button(
            label="📥 Download Processed Video",
            data=processed_video,
            file_name="crowd_analysis_output.mp4",
            mime="video/mp4",
            key="download_uploaded_processed_video",
        )

        # ====================================================
        # CSV REPORT
        # ====================================================

        csv_df = pd.DataFrame(
            {
                "Time (seconds)": time_data,
                "People Detected": people_data,
            }
        )

        st.download_button(
            label="📄 Download Crowd Report",
            data=csv_df.to_csv(
                index=False
            ),
            file_name="crowd_analysis_report.csv",
            mime="text/csv",
            key="download_uploaded_crowd_report",
        )

        # ====================================================
        # CLEANUP INPUT
        # ====================================================

        try:

            os.remove(
                input_path
            )

        except OSError:

            pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Crowd Management System • "
    "YOLO • Live Photo • Live Video • "
    "GPS • Location Text • Image Upload • "
    "Video Upload • HIGH-Crowd Email Alerts"
)
