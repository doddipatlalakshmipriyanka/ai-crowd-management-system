import io
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
st.set_page_config(page_title="AI Crowd Management System", page_icon="👥", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1000px;margin:auto;padding-top:2rem;padding-left:2rem;padding-right:2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SETTINGS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Crowd Threshold Settings")
crowd_threshold = st.sidebar.number_input(
    "🚨 High Crowd Threshold", min_value=1, max_value=500,
    value=20, step=1,
    help="Set the number of people above which the crowd is HIGH."
)
st.sidebar.info(f"🚨 HIGH crowd when people ≥ {crowd_threshold}")

CAMERA_CONFIDENCE = 0.25
CAMERA_IMAGE_SIZE = 416
HIGH_THRESHOLD = int(crowd_threshold)
HIGH_FRAMES_REQUIRED = 10
ALERT_COOLDOWN_SECONDS = 300
RECORD_DIR = Path("camera_records")
IMAGE_DIR = Path("image")
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "frame_store": {"frame": None, "lock": threading.Lock()},
        "camera_store": {"people": 0, "risk": "LOW", "high_frames": 0,
                          "last_alert": 0.0, "lock": threading.Lock()},
        "gps_store": {"latitude": None, "longitude": None, "accuracy": None,
                      "location_text": None},
        "record_store": {"record_id": str(uuid.uuid4())},
        "captured_photo": None,
        "captured_photo_people": 0,
        "captured_photo_risk": "LOW",
        "live_photo_alert_sent": False,
        "uploaded_image_alert_id": None,
        "uploaded_video_alert_id": None,
        "processed_video_id": None,
        "processed_video_bytes": None,
        "processed_video_report": None,
        "processed_video_max_people": 0,
        "processed_video_risk": "LOW",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()
frame_store = st.session_state.frame_store
camera_store = st.session_state.camera_store
gps_store = st.session_state.gps_store
record_id = st.session_state.record_store["record_id"]
record_file = RECORD_DIR / f"{record_id}_crowd_video.flv"

# ============================================================
# TITLE / BANNER
# ============================================================
st.title("👥 AI Crowd Management System")
st.write("AI-powered crowd detection, people counting, risk analysis, live photo/video capture and location-based HIGH-crowd alerts.")
st.markdown("---")
st.subheader("🏠 Welcome")

banner_path = next((p for p in [IMAGE_DIR / "banner (3).jpg", IMAGE_DIR / "banner (2).jpg", IMAGE_DIR / "banner.jpg"] if p.exists()), None)
if banner_path:
    st.image(str(banner_path), use_container_width=True)
else:
    st.warning("⚠️ Banner image not found. Put your banner image inside the image folder.")

st.write("This system detects people using YOLO and classifies crowd risk as LOW, MEDIUM or HIGH.")
c1, c2, c3 = st.columns(3)
with c1: st.info("📷 **Live Photo**\n\nCapture the current processed camera frame.")
with c2: st.info("🎥 **Live Video**\n\nRecord the live camera stream.")
with c3: st.info("🚨 **HIGH Alert**\n\nSend crowd count and location by email.")

# ============================================================
# YOLO
# ============================================================
@st.cache_resource
def load_model():
    path = Path("yolov8s.pt")
    if not path.exists():
        raise FileNotFoundError("yolov8s.pt was not found. Put it in the same folder as app.py.")
    return YOLO(str(path))

try:
    model = load_model()
except Exception as e:
    st.error("❌ YOLO model could not be loaded.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def get_risk(count, threshold):
    medium_threshold = max(1, int(threshold) // 2)
    if count < medium_threshold:
        return "LOW"
    if count < threshold:
        return "MEDIUM"
    return "HIGH"


def load_email_config():
    try:
        return {
            "sender_email": st.secrets.get("ALERT_EMAIL", ""),
            "sender_password": st.secrets.get("ALERT_PASSWORD", ""),
            "admin_email": st.secrets.get("ADMIN_EMAIL", ""),
        }
    except Exception:
        return {"sender_email": "", "sender_password": "", "admin_email": ""}

EMAIL_CONFIG = load_email_config()


def send_crowd_alert(people_count, location_text=None, latitude=None, longitude=None, source="Live Camera"):
    try:
        sender = EMAIL_CONFIG["sender_email"]
        password = EMAIL_CONFIG["sender_password"]
        admin = EMAIL_CONFIG["admin_email"]
        if not sender or not password or not admin:
            print("Email configuration is incomplete. Check ALERT_EMAIL, ALERT_PASSWORD and ADMIN_EMAIL.")
            return False

        if location_text:
            final_location = location_text
        elif latitude is not None and longitude is not None:
            final_location = f"Latitude: {float(latitude):.6f}\nLongitude: {float(longitude):.6f}"
        else:
            final_location = "GPS location unavailable."

        msg = EmailMessage()
        msg["Subject"] = "🚨 HIGH CROWD ALERT - AI Crowd Management System"
        msg["From"] = sender
        msg["To"] = admin
        msg.set_content(f"""HIGH CROWD ALERT
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
""")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("EMAIL SENT SUCCESSFULLY.")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print("Gmail authentication failed:", e)
        return False
    except Exception as e:
        print("Email alert error:", type(e).__name__, e)
        return False


def send_high_crowd_alert_once(people_count, source, alert_id=None):
    kwargs = {
        "people_count": people_count,
        "location_text": gps_store.get("location_text"),
        "latitude": gps_store.get("latitude"),
        "longitude": gps_store.get("longitude"),
        "source": source,
    }

    if source == "Live Photo Capture":
        if st.session_state.live_photo_alert_sent:
            return False
        ok = send_crowd_alert(**kwargs)
        if ok:
            st.session_state.live_photo_alert_sent = True
        return ok

    if source == "Uploaded Image":
        if alert_id is not None and st.session_state.uploaded_image_alert_id == alert_id:
            return False
        ok = send_crowd_alert(**kwargs)
        if ok:
            st.session_state.uploaded_image_alert_id = alert_id
        return ok

    if source == "Uploaded Video":
        if alert_id is not None and st.session_state.uploaded_video_alert_id == alert_id:
            return False
        ok = send_crowd_alert(**kwargs)
        if ok:
            st.session_state.uploaded_video_alert_id = alert_id
        return ok

    return send_crowd_alert(**kwargs)


def reverse_geocode(latitude, longitude):
    if latitude is None or longitude is None:
        return None
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": latitude, "lon": longitude, "format": "jsonv2", "zoom": 18, "addressdetails": 1},
            headers={"User-Agent": "AI-Crowd-Management-System/1.0"},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        if data.get("display_name"):
            return data["display_name"]
        address = data.get("address", {})
        parts = []
        for key in ["amenity", "building", "road", "suburb", "town", "city", "state", "postcode", "country"]:
            value = address.get(key)
            if value and value not in parts:
                parts.append(value)
        return ", ".join(parts) if parts else None
    except Exception as e:
        print("Reverse geocoding error:", e)
        return None

# ============================================================
# LIVE CAMERA CALLBACK
# ============================================================
def camera_frame_callback(frame: av.VideoFrame):
    image = frame.to_ndarray(format="bgr24")
    try:
        results = model(image, classes=[0], conf=CAMERA_CONFIDENCE, imgsz=CAMERA_IMAGE_SIZE, verbose=False)
    except Exception as e:
        print("YOLO camera error:", e)
        return frame

    people_count = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            try:
                if int(box.cls[0]) != 0:
                    continue
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "Person", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception:
                continue

    risk = get_risk(people_count, HIGH_THRESHOLD)
    with frame_store["lock"]:
        frame_store["frame"] = image.copy()

    should_alert = False
    with camera_store["lock"]:
        camera_store["people"] = people_count
        camera_store["risk"] = risk
        camera_store["high_frames"] = camera_store["high_frames"] + 1 if people_count >= HIGH_THRESHOLD else 0
        now = time.time()
        if camera_store["high_frames"] >= HIGH_FRAMES_REQUIRED and now - camera_store["last_alert"] >= ALERT_COOLDOWN_SECONDS:
            camera_store["last_alert"] = now
            should_alert = True

    if should_alert:
        threading.Thread(
            target=send_crowd_alert,
            kwargs={
                "people_count": people_count,
                "location_text": gps_store.get("location_text"),
                "latitude": gps_store.get("latitude"),
                "longitude": gps_store.get("longitude"),
                "source": "Live Camera",
            },
            daemon=True,
        ).start()

    color = (0, 255, 0) if risk == "LOW" else (0, 255, 255) if risk == "MEDIUM" else (0, 0, 255)
    cv2.putText(image, f"People: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, f"Risk: {risk}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, "GPS: ON" if gps_store.get("latitude") is not None else "GPS: OFF", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if gps_store.get("latitude") is not None else (0, 0, 255), 2)
    if risk == "HIGH":
        cv2.putText(image, "HIGH CROWD ALERT!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


def out_recorder_factory():
    return MediaRecorder(str(record_file), format="flv")

# ============================================================
# LIVE CAMERA UI
# ============================================================
st.markdown("---")
st.subheader("📷🎥 Live Camera")
st.write("Click START to activate your camera. The camera must be started before taking a live photo or recording live video.")

_, camera_center, _ = st.columns([1, 2, 1])
with camera_center:
    try:
        ctx = webrtc_streamer(
            key="crowd-live-camera-final",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=camera_frame_callback,
            out_recorder_factory=out_recorder_factory,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
    except Exception as e:
        st.error("❌ Camera could not be started.")
        st.code(f"{type(e).__name__}: {e}")
        ctx = None

camera_running = bool(ctx is not None and ctx.state.playing)
if camera_running:
    st.success("🟢 Camera is LIVE. YOLO detection and video recording are active.")
else:
    st.info("🔴 Camera is stopped. Click START above to activate it.")

# ============================================================
# LIVE PHOTO
# ============================================================
st.markdown("---")
st.subheader("📸 Live Photo Capture")
st.write("Start the camera first, wait for the live video to appear, then click **Capture Current Image**.")

if st.button("📸 Capture Current Image", key="capture_live_photo", disabled=not camera_running):
    st.session_state.live_photo_alert_sent = False
    with frame_store["lock"]:
        current_frame = None if frame_store["frame"] is None else frame_store["frame"].copy()
    if current_frame is None:
        st.warning("⚠️ Camera is running, but no processed frame is ready yet. Wait 1–2 seconds and try again.")
    else:
        with camera_store["lock"]:
            photo_people = camera_store["people"]
            photo_risk = camera_store["risk"]
        st.session_state.captured_photo = current_frame
        st.session_state.captured_photo_people = photo_people
        st.session_state.captured_photo_risk = photo_risk
        st.success("📸 Photo captured successfully!")

        if photo_risk == "HIGH":
            st.error(f"🚨 HIGH CROWD DETECTED! People: {photo_people}")
            ok = send_high_crowd_alert_once(photo_people, "Live Photo Capture")
            if ok:
                st.success("📧 HIGH crowd alert email sent successfully.")
            else:
                st.warning("⚠️ HIGH crowd detected, but the email alert could not be sent.")
        elif photo_risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM CROWD LEVEL. People: {photo_people}")
        else:
            st.success(f"✅ Crowd level is LOW. People: {photo_people}")

if st.session_state.captured_photo is not None:
    st.subheader("📸 Captured Live Photo")
    st.image(cv2.cvtColor(st.session_state.captured_photo, cv2.COLOR_BGR2RGB), caption="Captured Crowd Image", width=600)
    p1, p2 = st.columns(2)
    with p1: st.metric("👥 People Detected", st.session_state.captured_photo_people)
    with p2: st.metric("⚠️ Crowd Risk", st.session_state.captured_photo_risk)
    ok, enc = cv2.imencode(".jpg", st.session_state.captured_photo, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        st.download_button("📥 Download Captured Photo", enc.tobytes(), "live_crowd_photo.jpg", "image/jpeg", key="download_live_photo")

# ============================================================
# LIVE VIDEO
# ============================================================
st.markdown("---")
st.subheader("🎥 Live Video Capture")
if camera_running:
    st.info("🎥 Live video recording is active. Keep the camera running and click STOP when you want to finish recording.")
elif record_file.exists() and record_file.stat().st_size > 0:
    try:
        video_bytes = record_file.read_bytes()
        st.success("✅ Live crowd video recording completed.")
        try:
            st.video(video_bytes)
        except Exception:
            st.info("ℹ️ FLV inline playback may not be supported by your browser. Download the file instead.")
        st.download_button("📥 Download Live Crowd Video", video_bytes, "live_crowd_video.flv", "video/x-flv", key="download_live_crowd_video")
    except Exception as e:
        st.error("❌ Video file could not be read.")
        st.code(f"{type(e).__name__}: {e}")
else:
    st.info("📹 No recorded video yet. Start the camera, keep it running, then click STOP.")

# ============================================================
# CAMERA STATUS
# ============================================================
st.markdown("---")
st.subheader("📊 Live Camera Status")
with camera_store["lock"]:
    current_people = camera_store["people"]
    current_risk = camera_store["risk"]
s1, s2, s3 = st.columns(3)
with s1: st.metric("👥 People Detected", current_people)
with s2: st.metric("⚠️ Current Risk", current_risk)
with s3: st.success("📍 GPS Active") if gps_store["latitude"] is not None else st.warning("📍 GPS Unavailable")
if current_risk == "HIGH": st.error("🚨 HIGH CROWD DETECTED!")

# ============================================================
# GPS
# ============================================================
st.markdown("---")
st.subheader("📍 GPS Location")
st.write("GPS is independent from the camera. Click GET GPS LOCATION and allow browser location permission.")
gps_result = get_geolocation("📍 GET GPS LOCATION")

if isinstance(gps_result, dict):
    if "error" in gps_result:
        error = gps_result.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message", "Unknown GPS error") if isinstance(error, dict) else str(error)
        if code == 1: st.error("❌ Browser location permission was denied.")
        elif code == 2: st.warning("⚠️ Location is temporarily unavailable.")
        elif code == 3: st.warning("⚠️ GPS request timed out. Click GET GPS LOCATION again.")
        else: st.warning(f"⚠️ GPS error: {message}")
    elif gps_result.get("coords"):
        coords = gps_result["coords"]
        lat, lon = coords.get("latitude"), coords.get("longitude")
        accuracy = coords.get("accuracy")
        if lat is not None and lon is not None:
            gps_store["latitude"] = float(lat)
            gps_store["longitude"] = float(lon)
            gps_store["accuracy"] = float(accuracy) if accuracy is not None else None
            gps_store["location_text"] = reverse_geocode(float(lat), float(lon)) or "Location name could not be determined"

latitude, longitude = gps_store["latitude"], gps_store["longitude"]
if latitude is not None and longitude is not None:
    st.success("📍 GPS location detected successfully.")
    st.info(f"📍 **{gps_store['location_text']}**")
    if gps_store["accuracy"] is not None:
        st.caption(f"GPS accuracy: approximately ±{gps_store['accuracy']:.1f} m")
    gps_map = folium.Map(location=[latitude, longitude], zoom_start=16)
    folium.Marker([latitude, longitude], popup=gps_store["location_text"], tooltip="📍 Crowd Monitoring Location").add_to(gps_map)
    st_folium(gps_map, width=700, height=350, key="current_gps_map")
else:
    st.info("📍 GPS location has not been detected yet. Click GET GPS LOCATION and allow browser location permission.")

# ============================================================
# IMAGE UPLOAD
# ============================================================
st.markdown("---")
st.subheader("📁 Upload Crowd Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="crowd_image_upload")

if uploaded_image is not None:
    image_alert_id = f"{uploaded_image.name}_{uploaded_image.size}"
    image = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("❌ Unable to read image.")
    else:
        results = model.predict(source=image, classes=[0], conf=0.15, iou=0.50, imgsz=960, max_det=1000, verbose=False)
        people_count = 0
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                if int(box.cls[0]) != 0: continue
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Person {confidence:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        risk = get_risk(people_count, crowd_threshold)
        st.subheader("📷 Image Crowd Analysis")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Crowd", width=600)
        u1, u2 = st.columns(2)
        with u1: st.metric("👥 People Detected", people_count)
        with u2: st.metric("⚠️ Crowd Risk", risk)

        if risk == "HIGH":
            st.error(f"🚨 HIGH CROWD DETECTED! People: {people_count}")
            st.warning(f"📍 Alert Location: {gps_store['location_text'] or 'GPS unavailable'}")
            if st.session_state.uploaded_image_alert_id != image_alert_id:
                ok = send_high_crowd_alert_once(people_count, "Uploaded Image", image_alert_id)
                st.success("📧 HIGH crowd alert email sent successfully.") if ok else st.warning("⚠️ HIGH crowd detected, but the email alert could not be sent.")
            else:
                st.info("📧 Alert already sent for this uploaded image.")
        elif risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM CROWD LEVEL. People: {people_count}")
        else:
            st.success(f"✅ Crowd level is LOW. People: {people_count}")

# ============================================================
# VIDEO UPLOAD
# ============================================================
st.markdown("---")
st.subheader("🎥 Upload Crowd Video")
uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="crowd_video_upload")

if uploaded_video is not None:
    video_alert_id = f"{uploaded_video.name}_{uploaded_video.size}"

    # Prevent expensive video processing on every Streamlit rerun.
    if st.session_state.processed_video_id != video_alert_id:
        input_path = None
        output_path = None
        try:
            suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_video.getvalue())
                input_path = f.name

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("❌ Unable to open video.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not out.isOpened(): raise RuntimeError("OpenCV could not create the output video.")

                time_data, people_data = [], []
                max_people = 0
                frame_number = 0
                last_people_count = 0
                progress = st.progress(0)
                status = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    if frame_number % 2 == 0:
                        results = model.predict(source=frame, classes=[0], conf=0.15, iou=0.50, imgsz=640, max_det=1000, verbose=False)
                        people_count = 0
                        for result in results:
                            if result.boxes is None: continue
                            for box in result.boxes:
                                if int(box.cls[0]) != 0: continue
                                people_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                confidence = float(box.conf[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Person {confidence:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        last_people_count = people_count
                    else:
                        people_count = last_people_count

                    max_people = max(max_people, people_count)
                    risk = get_risk(people_count, crowd_threshold)
                    color = (0, 255, 0) if risk == "LOW" else (0, 255, 255) if risk == "MEDIUM" else (0, 0, 255)
                    cv2.putText(frame, f"People: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Risk: {risk}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    if risk == "HIGH": cv2.putText(frame, "HIGH CROWD ALERT!", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    out.write(frame)

                    time_data.append(round(frame_number / fps, 2))
                    people_data.append(people_count)
                    frame_number += 1
                    if total_frames > 0:
                        pct = min(frame_number / total_frames, 1.0)
                        progress.progress(pct)
                        status.write(f"Processing video... {int(pct * 100)}% | People: {people_count} | Risk: {risk}")

                cap.release(); out.release(); progress.empty(); status.empty()
                processed_video = Path(output_path).read_bytes()
                report_df = pd.DataFrame({"Time (seconds)": time_data, "People Detected": people_data})
                st.session_state.processed_video_id = video_alert_id
                st.session_state.processed_video_bytes = processed_video
                st.session_state.processed_video_report = report_df.to_csv(index=False)
                st.session_state.processed_video_max_people = max_people
                st.session_state.processed_video_risk = get_risk(max_people, crowd_threshold)
                st.success("🎉 Video processing completed!")
        except Exception as e:
            st.error("❌ Error while processing uploaded video.")
            st.code(f"{type(e).__name__}: {e}")
        finally:
            if input_path:
                try: os.remove(input_path)
                except OSError: pass
            if output_path:
                try: os.remove(output_path)
                except OSError: pass

    if st.session_state.processed_video_id == video_alert_id and st.session_state.processed_video_bytes:
        processed_video = st.session_state.processed_video_bytes
        max_people = st.session_state.processed_video_max_people
        final_risk = st.session_state.processed_video_risk
        report_csv = st.session_state.processed_video_report

        st.subheader("🎬 Processed Video")
        st.video(processed_video)
        st.subheader("📊 Video Crowd Analysis")
        v1, v2 = st.columns(2)
        with v1: st.metric("👥 Maximum People Detected", max_people)
        with v2: st.metric("⚠️ Maximum Crowd Risk", final_risk)

        if final_risk == "HIGH":
            st.error(f"🚨 HIGH CROWD DETECTED! Maximum people: {max_people}")
            st.warning(f"📍 Alert Location: {gps_store['location_text'] or 'GPS unavailable'}")
            if st.session_state.uploaded_video_alert_id != video_alert_id:
                ok = send_high_crowd_alert_once(max_people, "Uploaded Video", video_alert_id)
                st.success("📧 HIGH crowd alert email sent successfully.") if ok else st.warning("⚠️ HIGH crowd detected, but the email alert could not be sent.")
            else:
                st.info("📧 Alert already sent for this uploaded video.")
        elif final_risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM CROWD LEVEL. Maximum people: {max_people}")
        else:
            st.success(f"✅ Crowd level is LOW. Maximum people: {max_people}")

        if report_csv:
            graph_df = pd.read_csv(io.StringIO(report_csv))
            st.subheader("📈 Crowd Count Over Time")
            st.line_chart(graph_df, x="Time (seconds)", y="People Detected")

        st.download_button("📥 Download Processed Video", processed_video, "crowd_analysis_output.mp4", "video/mp4", key=f"download_video_{video_alert_id}")
        st.download_button("📄 Download Crowd Report", report_csv, "crowd_analysis_report.csv", "text/csv", key=f"download_report_{video_alert_id}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("AI Crowd Management System • YOLO • Live Photo • Live Video • GPS • Location Text • Image Upload • Video Upload • HIGH-Crowd Email Alerts")
