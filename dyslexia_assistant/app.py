import streamlit as st
import speech_recognition as sr
import numpy as np
import cv2
import time
import pyttsx3
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------------------------------------------------
# Global setup
# -------------------------------------------------
st.set_page_config(page_title="AI Dyslexia Reading Assistant", layout="wide")
engine = pyttsx3.init()
DATA_FILE = Path("reading_sessions.csv")

LEVELS = ["Easy", "Medium", "Hard"]

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


def highlight_word_differences(expected: str, actual: str) -> str:
    """
    Show word-level feedback:
    - correct words in green
    - missed words in red with underline
    """
    expected_words = expected.split()
    actual_words = actual.split()

    actual_lower = [w.lower() for w in actual_words]

    html_chunks = []
    for w in expected_words:
        if w.lower() in actual_lower:
            # correct / present
            html_chunks.append(
                f"<span style='color:#2ecc71; font-weight:600; margin-right:4px;'>{w}</span>"
            )
        else:
            # missed / mispronounced
            html_chunks.append(
                f"<span style='color:#e74c3c; text-decoration:underline; margin-right:4px;'>{w}</span>"
            )

    return "<div style='font-size:18px; line-height:1.8;'>" + " ".join(html_chunks) + "</div>"


def analyze_attention(duration_seconds: int = 8) -> dict:
    """Face + eye detection → focus %, blink rate, rough gaze distribution."""
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
        roi_gray = gray[y:y + h, x:x + w]

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
    Simple regression-style score combining pronunciation, focus, and reading speed.
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


def rule_based_next_level(perf_idx: float, current_level: str = "Medium") -> str:
    """Original threshold-based suggestion (fallback if not enough data)."""
    levels = LEVELS
    i = levels.index(current_level)
    if perf_idx > 85 and i < 2:
        i += 1
    elif perf_idx < 60 and i > 0:
        i -= 1
    return levels[i]


def learned_next_level(
    pron: float, focus: float, speed_wps: float, current_level: str, history: pd.DataFrame
) -> str:
    """
    ML-style difficulty suggestion that uses historical engagement by level.
    Target is to keep engagement around ~80.
    """
    if history is None or history.empty or "engagement" not in history.columns:
        return rule_based_next_level(performance_index(pron, focus, speed_wps), current_level)

    if len(history) < 5:
        # not enough data to "learn" – use rule-based
        return rule_based_next_level(performance_index(pron, focus, speed_wps), current_level)

    current_eng = engagement_score(pron, focus, speed_wps)
    target = 80.0

    # average engagement per level from history
    level_means = history.groupby("level_used")["engagement"].mean()

    best_level = current_level
    best_diff = float("inf")

    for lvl in LEVELS:
        expected = level_means.get(lvl, current_eng)  # if no past data for that level
        diff = abs(expected - target)

        # do not jump more than one step at a time
        if abs(LEVELS.index(lvl) - LEVELS.index(current_level)) > 1:
            continue

        if diff < best_diff:
            best_diff = diff
            best_level = lvl

    return best_level


def text_to_speech(text: str):
    engine.say(text)
    engine.runAndWait()


def highlight_syllables(text: str, font_size: int = 22, line_height: float = 1.6) -> str:
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
    spans = ""
    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        spans += (
            f"<span style='background-color:{color}; padding:2px 4px; "
            f"border-radius:4px; margin-right:2px;'>{chunk}</span>"
        )
    wrapper = (
        f"<div style='font-size:{font_size}px; line-height:{line_height};'>"
        f"{spans}</div>"
    )
    return wrapper


def mentor_message(
    pron, focus, blink_rate, perf_idx, engagement, trend, persona: str
) -> str:
    """Generate adaptive mentor feedback based on metrics + persona style."""

    improving = trend == "up"
    declining = trend == "down"

    if persona == "Friendly Buddy":
        if improving:
            return (
                "🌟 You're on a roll! Your reading is getting better each time. "
                "Let's keep this streak and try a fun, slightly harder sentence next. "
            )
        if declining:
            return (
                "I can see today is a bit tougher, and that's totally okay. "
                "Let's take a small break, breathe, and then we’ll read it together again 😊"
            )
        if pron > 85 and focus > 80:
            return (
                "Amazing job! You stayed focused and your pronunciation was super clear. "
                "I'm really proud of you! 💚"
            )
        if focus < 60 or blink_rate > 25:
            return (
                "Looks like your eyes are getting tired. How about a quick stretch "
                "and some water before we try again? 🧃"
            )
        return "Nice effort! Every try makes you stronger at reading. 💪"

    if persona == "Supportive Coach":
        if improving:
            return (
                "Great progress – your engagement trend is going up. "
                "We can safely increase the challenge a bit while keeping you supported. 🏅"
            )
        if declining:
            return (
                "Your recent sessions show a drop in engagement. "
                "Let's temporarily lower difficulty, focus on accuracy, and rebuild confidence."
            )
        if perf_idx > 85:
            return (
                "You mastered this level. I recommend moving to the next difficulty tier "
                "to keep your brain positively challenged."
            )
        if pron < 60:
            return (
                "Many words were tricky this time. We'll slow down, repeat key words, "
                "and use text-to-speech to model the pronunciation."
            )
        return "Solid performance. Maintain this level for a few more attempts, then we’ll reassess."

    # Calm Guide
    if improving:
        return (
            "Your scores show gentle improvement over time. "
            "You are moving in the right direction – steady and calm. 🌱"
        )
    if declining:
        return (
            "Your recent engagement is slightly lower. "
            "Let's not worry – we can take things slowly and focus on one short sentence at a time."
        )
    if pron > 85 and focus > 80:
        return (
            "You read this sentence with clarity and attention. "
            "We can softly transition to more complex sentences when you feel ready."
        )
    return (
        "Thank you for your effort. With a bit more practice on the highlighted words "
        "and relaxed breathing, your reading will feel smoother."
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
    df = pd.read_csv(DATA_FILE)
    # ensure engagement column exists for older files
    if "engagement" not in df.columns:
        df["engagement"] = np.nan
    return df


def save_session(row: dict):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)


def engagement_trend(history: pd.DataFrame) -> str:
    """Return 'up', 'down', or 'flat' based on last few engagement scores."""
    if history is None or history.empty or "engagement" not in history.columns:
        return "flat"
    recent = history["engagement"].dropna().tail(5)
    if len(recent) < 3:
        return "flat"
    first = recent.iloc[0]
    last = recent.iloc[-1]
    if last - first > 5:
        return "up"
    if first - last > 5:
        return "down"
    return "flat"


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📚 AI Reading Assistant for Dyslexic Learners")

recognizer = sr.Recognizer()
history = load_history()
prev_session = history.tail(1) if not history.empty else pd.DataFrame()

audio_score = 0.0
reading_speed_wps = 1.0
focus_score = 0.0
blink_rate = 0.0
perf_idx = 0.0
engagement = 0.0
has_attention_scan = False

# Mentor persona selection
persona = st.sidebar.selectbox(
    "Mentor persona style",
    ["Friendly Buddy", "Supportive Coach", "Calm Guide"],
)

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

            st.markdown("**Word-by-word feedback:**", unsafe_allow_html=True)
            st.markdown(
                highlight_word_differences(expected_text, actual_text),
                unsafe_allow_html=True,
            )

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

# ML-powered suggestion
ml_suggested = learned_next_level(
    audio_score, effective_focus, reading_speed_wps, current_level_default, history
)

levels = list(sentences.keys())
selected_level = st.selectbox(
    "Difficulty level (ML suggestion can be overridden by teacher):",
    levels,
    index=levels.index(ml_suggested),
)
st.caption(
    f"Rule-based index = {perf_idx} → ML-suggested level based on past engagement: "
    f"**{ml_suggested}**."
)

# Multisensory controls
st.write("📖 Read this sentence (multisensory support enabled):")
font_size = st.slider("Font size", min_value=18, max_value=36, value=22, step=1)
line_height = st.slider("Line spacing", min_value=1.2, max_value=2.0, value=1.6, step=0.1)

current_sentence = sentences[selected_level]
st.markdown(
    highlight_syllables(current_sentence, font_size=font_size, line_height=line_height),
    unsafe_allow_html=True,
)

st.caption(
    f"Sentence length: {len(current_sentence.split())} words, "
    f"approx. {len(current_sentence)//3 + 1} syllable groups."
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
    "Instant support: colour cues + speech model help the learner match what they "
    "see with what they hear, ideal for dyslexic readers."
)

# -------------------- STEP 5: MENTOR -------------------- #
st.subheader("Step 5️⃣ Mentor – AI Persona & Motivational Feedback")

trend = engagement_trend(history)

if audio_score > 0:
    if not has_attention_scan:
        st.info(
            "Tip: For more accurate coaching, run the 8-second attention scan "
            "while you are reading the sentence."
        )

    msg = mentor_message(
        audio_score, effective_focus, blink_rate, perf_idx, engagement, trend, persona
    )
    st.success(msg)

    # Short session summary compared to previous session
    if not prev_session.empty:
        prev_pron = float(prev_session["pron_score"].iloc[0])
        prev_eng = float(prev_session["engagement"].iloc[0]) if not np.isnan(prev_session["engagement"].iloc[0]) else engagement
        delta_pron = round(audio_score - prev_pron, 1)
        delta_eng = round(engagement - prev_eng, 1)

        summary_lines = []

        if abs(delta_pron) >= 1:
            if delta_pron > 0:
                summary_lines.append(f"✅ Pronunciation improved by {delta_pron} points since last session.")
            else:
                summary_lines.append(f"ℹ️ Pronunciation dropped by {abs(delta_pron)} points; we’ll reinforce tricky words.")

        if abs(delta_eng) >= 1:
            if delta_eng > 0:
                summary_lines.append(f"✅ Engagement increased by {delta_eng} points – great focus!")
            else:
                summary_lines.append(f"ℹ️ Engagement decreased by {abs(delta_eng)} points; maybe take a short break.")

        if summary_lines:
            st.caption("Session summary:\n- " + "\n- ".join(summary_lines))

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
