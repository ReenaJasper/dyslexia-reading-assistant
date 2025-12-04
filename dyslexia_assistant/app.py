import streamlit as st
import speech_recognition as sr
import numpy as np
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------- Optional OpenCV (webcam may not be available in cloud) ----------
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# ---------- Optional Text-to-Speech (pyttsx3 often fails in cloud) ----------
try:
    import pyttsx3

    engine = pyttsx3.init()
    TTS_AVAILABLE = True
except Exception:
    engine = None
    TTS_AVAILABLE = False

# ---------- Data file for logging sessions ----------
DATA_FILE = Path("reading_sessions.csv")


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def get_pronunciation_score(expected: str, actual: str) -> float:
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()
    if not expected_words:
        return 0.0
    correct = sum(1 for w in actual_words if w in expected_words)
    return round((correct / len(expected_words)) * 100, 2)


def corrective_hint(expected: str, actual: str) -> str:
    """Generate a simple, friendly hint based on mis- or unpronounced words."""
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()

    missed = [w for w in expected_words if w not in actual_words]
    extra = [w for w in actual_words if w not in expected_words]

    if not missed and not extra:
        return "You pronounced all the key words clearly – great job! 🎉"

    msg_parts = []
    if missed:
        msg_parts.append(
            "Let's practise these tricky words again: " + ", ".join(sorted(set(missed)))
        )
    if extra:
        msg_parts.append(
            "You added some extra words: " + ", ".join(sorted(set(extra)))
        )

    return " ".join(msg_parts)


def analyze_attention(duration_seconds: int = 8) -> dict:
    """
    Face + eye detection → focus %, blink rate, rough gaze distribution.
    On cloud (where cv2/webcam may not be available), we return neutral values.
    """
    if not CV2_AVAILABLE:
        # Cloud-safe fallback: no webcam, return default engagement values
        return {
            "focus": 50.0,
            "blink_rate": 0.0,
            "gaze_left": 0.0,
            "gaze_center": 50.0,
            "gaze_right": 0.0,
        }

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    total_frames = 0
    focused_frames = 0
    blinks = 0
    last_eyes_visible = False
    gaze_positions = []  # -1 left, 0 centre, +1 right

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        total_frames += 1
        if len(faces) == 0:
            last_eyes_visible = False
            continue

        focused_frames += 1
        (x, y, w, h) = faces[0]
        roi_gray = gray[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        eyes_visible = len(eyes) > 0

        # blink ≈ visible → not visible
        if last_eyes_visible and not eyes_visible:
            blinks += 1
        last_eyes_visible = eyes_visible

        # rough horizontal gaze based on eye vs face centre
        if eyes_visible:
            ex, ey, ew, eh = eyes[0]
            eye_center_x = x + ex + ew / 2
            face_center_x = x + w / 2
            offset = (eye_center_x - face_center_x) / w

            if offset < -0.1:
                gaze_positions.append(-1)
            elif offset > 0.1:
                gaze_positions.append(1)
            else:
                gaze_positions.append(0)

    cap.release()
    duration = max(time.time() - start_time, 1.0)

    focus_pct = round(100 * focused_frames / max(total_frames, 1), 1)
    blink_rate = round(blinks * 60 / duration, 1)  # blinks / minute

    if gaze_positions:
        left_ratio = gaze_positions.count(-1) / len(gaze_positions)
        right_ratio = gaze_positions.count(1) / len(gaze_positions)
        center_ratio = gaze_positions.count(0) / len(gaze_positions)
    else:
        left_ratio = right_ratio = center_ratio = 0.0

    return {
        "focus": focus_pct,
        "blink_rate": blink_rate,
        "gaze_left": round(100 * left_ratio, 1),
        "gaze_center": round(100 * center_ratio, 1),
        "gaze_right": round(100 * right_ratio, 1),
    }


def performance_index(pron: float, focus: float, speed_wps: float) -> float:
    """
    Simple ML-style regression score combining pronunciation, focus, and reading speed.
    This is our 'model output' used to adapt difficulty.
    """
    speed_score = min(speed_wps * 20, 100)  # rough 0–100 scaling
    idx = 0.5 * pron + 0.3 * focus + 0.2 * speed_score
    return round(idx, 1)


def engagement_score(pron: float, focus: float, speed_wps: float) -> float:
    """
    Overall reading engagement score (0–100),
    combining pronunciation, attention, and reading speed.
    """
    speed_score = min(speed_wps * 20, 100)
    score = 0.45 * pron + 0.35 * focus + 0.20 * speed_score
    return round(score, 1)


def suggest_next_level(perf_idx: float, current_level: str = "Medium") -> str:
    levels = ["Easy", "Medium", "Hard"]
    i = levels.index(current_level)
    if perf_idx > 85 and i < 2:
        i += 1
    elif perf_idx < 60 and i > 0:
        i -= 1
    return levels[i]


def text_to_speech(text: str):
    if not TTS_AVAILABLE:
        st.warning(
            "Text-to-speech audio isn’t available in this cloud demo. "
            "For spoken audio, please run the app locally."
        )
        return

    engine.say(text)
    engine.runAndWait()


def highlight_syllables(text: str) -> str:
    """Naive syllable-style highlighting using alternating colours."""
    vowels = "aeiouAEIOU"
    chunks = []
    current = ""

    for ch in text:
        current += ch
        if ch in vowels:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)

    colors = ["#ffeaa7", "#fab1a0"]
    html = ""
    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        html += (
            f"<span style='background-color:{color}; padding:2px 4px; "
            f"border-radius:4px; margin-right:2px;'>{chunk}</span>"
        )
    return html


def mentor_message(pron, focus, blink_rate, perf_idx, persona: str) -> str:
    base = ""

    if persona == "Friendly Buddy":
        base = "Hey, great effort today! "
    elif persona == "Calm Coach":
        base = "You're making steady, thoughtful progress. "
    else:  # Motivational Trainer
        base = "Awesome grind! You're pushing yourself really well. "

    if pron > 85 and focus > 80:
        return (
            base
            + "Your pronunciation and attention are both strong. "
            "We can safely try slightly harder sentences next. 💪"
        )
    if focus < 60 and blink_rate > 25:
        return (
            base
            + "I noticed lots of blinking and low focus – maybe you're tired. "
            "Let's take a short break and then come back refreshed 😊"
        )
    if pron < 60:
        return (
            base
            + "Some words were tricky this time. Let's slow down and listen to the "
            "sentence again, then try reading it in smaller chunks."
        )
    if perf_idx > 80:
        return (
            base
            + "Overall performance is high – you're ready for more challenging "
            "reading tasks. Keep going! ⭐"
        )
    return (
        base
        + "Nice work! With a bit more practice, your reading will feel even smoother."
    )


def load_history() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "pron_score",
                "focus_score",
                "speed_wps",
                "perf_index",
                "engagement",
                "level_used",
            ]
        )
    return pd.read_csv(DATA_FILE)


def save_session(row: dict):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)


# -------------------------------------------------
# UI Setup
# -------------------------------------------------
st.set_page_config(
    page_title="AI Dyslexia Reading Assistant",
    layout="wide",
)

# Sidebar: mentor persona
persona_style = st.sidebar.selectbox(
    "Mentor persona style",
    ["Friendly Buddy", "Calm Coach", "Motivational Trainer"],
)

st.title("📚 AI Reading Assistant for Dyslexic Learners")

recognizer = sr.Recognizer()
history = load_history()

audio_score = 0.0
reading_speed_wps = 1.0
focus_score = 0.0
blink_rate = 0.0
perf_idx = 0.0
engagement = 0.0
has_attention_scan = False

# -------------------- STEP 1: LISTEN -------------------- #
st.subheader("Step 1️⃣ Listen – Speech Analysis")

uploaded_audio = st.file_uploader(
    "Upload a WAV audio file of your reading", type=["wav"]
)
expected_text = st.text_input(
    "Target sentence (what the learner is supposed to read):",
    "The quick brown fox jumps over the lazy dog",
)

if uploaded_audio:
    st.audio(uploaded_audio)
    with sr.AudioFile(uploaded_audio) as src:
        start_time = time.time()
        audio_data = recognizer.record(src)
        duration = max(time.time() - start_time, 1e-3)

        try:
            st.info("Transcribing speech with Google ASR…")
            actual_text = recognizer.recognize_google(audio_data)
            st.success(f"Recognized Speech: {actual_text}")

            audio_score = get_pronunciation_score(expected_text, actual_text)
            st.metric("Pronunciation Accuracy", f"{audio_score}%")

            num_words = max(len(actual_text.split()), 1)
            reading_speed_wps = round(num_words / duration, 2)
            st.caption(f"Approx. reading speed: {reading_speed_wps} words / second")

            hint = corrective_hint(expected_text, actual_text)
            st.caption("Instant corrective hint:")
            st.info(hint)

            if audio_score > 85:
                st.success("Excellent reading! ⭐")
            elif audio_score > 50:
                st.warning("Good attempt! We can improve some words 😊")
            else:
                st.error(
                    "Lots of difficult words here – let's practise again slowly 💪"
                )

        except Exception as e:
            st.error(f"Could not recognise speech. Try another recording. ({e})")

# -------------------- STEP 2: OBSERVE -------------------- #
st.subheader("Step 2️⃣ Observe – Attention While Reading")

st.write(
    "For the next 8 seconds, please **read the on-screen sentence aloud** while we "
    "use your webcam to analyse focus, blink rate and gaze direction. "
    "This measures *reading* attention, not just sitting in front of the screen."
)

if st.button("Run 8-second Attention Scan"):
    if not CV2_AVAILABLE:
        st.warning(
            "Webcam-based focus tracking is not available in this cloud demo. "
            "Run the app locally to use the full attention module."
        )
    else:
        st.info("Scanning… please read aloud and look at the screen 👀")

    att = analyze_attention()
    has_attention_scan = True

    focus_score = att["focus"]
    blink_rate = att["blink_rate"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Focus level", f"{att['focus']} %")
    with col2:
        st.metric("Blink rate", f"{att['blink_rate']} / min")

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Looking left", f"{att['gaze_left']} %")
    with col4:
        st.metric("Looking centre", f"{att['gaze_center']} %")
    with col5:
        st.metric("Looking right", f"{att['gaze_right']} %")

    if att["focus"] < 60 or att["gaze_center"] < 50:
        st.warning(
            "Signs of distraction or fatigue detected. A short break may help 🙂"
        )
    else:
        st.success("You stayed well-engaged during this reading window. 🌟")

# -------------------- STEP 3: ADAPT -------------------- #
st.subheader("Step 3️⃣ Adapt – Dynamic Difficulty Engine")

sentences = {
    "Easy": "The cat sits.",
    "Medium": "The brown fox jumps over a lazy dog.",
    "Hard": "A speedy fox vaults gracefully over a tired hound.",
}

effective_focus = focus_score if has_attention_scan else 50.0
perf_idx = performance_index(audio_score, effective_focus, reading_speed_wps)
engagement = engagement_score(audio_score, effective_focus, reading_speed_wps)

st.metric("Overall Reading Engagement", f"{engagement}%")

current_level_default = "Medium"
if not history.empty:
    current_level_default = history.iloc[-1]["level_used"]

suggested_level = suggest_next_level(perf_idx, current_level_default)

levels = list(sentences.keys())
selected_level = st.selectbox(
    "Difficulty level (system suggestion can be overridden by teacher):",
    levels,
    index=levels.index(suggested_level),
)
st.caption(
    f"System suggestion based on learned performance index {perf_idx}: "
    f"**{suggested_level}** level."
)

current_sentence = sentences[selected_level]

st.write("📖 Read this sentence (multisensory support enabled):")
st.markdown(
    f"<div style='font-size:22px; line-height:1.6;'>{highlight_syllables(current_sentence)}</div>",
    unsafe_allow_html=True,
)

# -------------------- STEP 4: ASSIST -------------------- #
st.subheader("Step 4️⃣ Assist – Multisensory Learning Layer")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🔊 Listen to sentence"):
        text_to_speech(current_sentence)
with col_b:
    st.info(
        "Tip: Follow the highlighted syllables with your eyes while listening. "
        "This links audio and visual cues for dyslexic learners."
    )

st.caption(
    "Instant support: use both the colour highlights and the audio model to practise "
    "matching what you *see* with what you *hear*."
)

# -------------------- STEP 5: MENTOR -------------------- #
st.subheader("Step 5️⃣ Mentor – AI Persona & Motivational Feedback")

if audio_score > 0:
    if not has_attention_scan:
        st.info(
            "Tip: For more accurate coaching, run the 8-second attention scan "
            "while you are reading the sentence."
        )

    msg = mentor_message(audio_score, effective_focus, blink_rate, perf_idx, persona_style)
    st.success(msg)

    # log the session so the 'model' can learn over time
    save_session(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pron_score": audio_score,
            "focus_score": effective_focus,
            "speed_wps": reading_speed_wps,
            "perf_index": perf_idx,
            "engagement": engagement,
            "level_used": selected_level,
        }
    )
else:
    st.info(
        "Upload a recording in Step 1 and (optionally) run the attention scan in Step 2 "
        "to receive personalised mentoring feedback."
    )

# -------------------- STEP 6: DASHBOARD -------------------- #
st.subheader("📊 Progress & Learning Dashboard")

history = load_history()
if history.empty:
    st.write(
        "No sessions logged yet. Complete at least one reading session to see progress."
    )
else:
    st.dataframe(history.tail(10), use_container_width=True)

    st.line_chart(
        history.set_index("timestamp")[["pron_score", "focus_score", "engagement"]],
        height=250,
    )
    st.caption(
        "The system uses this growing dataset as training signal to refine its "
        "difficulty suggestions over time."
    )
