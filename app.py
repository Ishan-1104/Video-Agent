import os
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extract import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme Definitions ───────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#0a0a0f", "surface": "#111118", "surface2": "#1a1a25",
        "border": "#2a2a3a", "text": "#e8e8f0", "text_muted": "#7070a0",
        "accent": "#7c3aed", "accent_glow": "#9f67ff", "accent2": "#06b6d4",
        "success": "#10b981", "warning": "#f59e0b", "danger": "#ef4444",
        "grid_op": "0.03",
    },
    "light": {
        "bg": "#f4f4f9", "surface": "#ffffff", "surface2": "#f0f0f7",
        "border": "#dcdce8", "text": "#191926", "text_muted": "#6a6a85",
        "accent": "#7c3aed", "accent_glow": "#6d28d9", "accent2": "#0891b2",
        "success": "#059669", "warning": "#b45309", "danger": "#dc2626",
        "grid_op": "0.05",
    },
}

STEPS = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️", "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
]

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

# ─── Session State Init ──────────────────────────────────────────────────────────
DEFAULTS = {
    "sessions": [],          # list of completed session dicts
    "active_idx": None,      # index into sessions currently displayed
    "processing": False,
    "pipeline_steps": {},
    "step_times": {},
    "theme": "dark",
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css(t: dict):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

    :root {{
        --bg: {t['bg']};
        --surface: {t['surface']};
        --surface-2: {t['surface2']};
        --border: {t['border']};
        --accent: {t['accent']};
        --accent-glow: {t['accent_glow']};
        --accent-2: {t['accent2']};
        --text: {t['text']};
        --text-muted: {t['text_muted']};
        --success: {t['success']};
        --warning: {t['warning']};
        --danger: {t['danger']};
    }}

    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', monospace;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }}

    .stApp {{ background: var(--bg) !important; }}

    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            linear-gradient(rgba(124, 58, 237, {t['grid_op']}) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124, 58, 237, {t['grid_op']}) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }}

    [data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }}
    [data-testid="stSidebar"] * {{ color: var(--text) !important; }}

    h1, h2, h3, h4, h5, h6 {{ font-family: 'Syne', sans-serif !important; color: var(--text) !important; }}

    .hero-title {{
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
        background: linear-gradient(135deg, var(--text) 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .hero-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-muted);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }}

    .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    }}
    .card:hover {{ border-color: var(--accent); }}
    .card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: linear-gradient(180deg, var(--accent), var(--accent-2));
    }}

    .card-title {{
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .card-content {{ font-size: 0.875rem; line-height: 1.7; color: var(--text); }}

    .badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}
    .badge-purple {{ background: rgba(124,58,237,0.18); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }}
    .badge-cyan   {{ background: rgba(6,182,212,0.13);  color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }}
    .badge-green  {{ background: rgba(16,185,129,0.13); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }}

    .stTextInput > div > div > input,
    .stSelectbox > div > div {{
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
        text-transform: uppercase !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.35) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }}
    .stButton > button:disabled {{
        opacity: 0.5 !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    .followup-btn button {{
        background: var(--surface-2) !important;
        border: 1px dashed var(--accent) !important;
        color: var(--accent-glow) !important;
        font-size: 0.72rem !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 0.4rem 0.8rem !important;
    }}

    .status-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.65rem 1rem;
        background: var(--surface-2);
        border-radius: 8px;
        margin: 0.4rem 0;
        border: 1px solid var(--border);
        font-size: 0.78rem;
    }}
    .status-bar-left {{ display: flex; align-items: center; gap: 0.6rem; }}
    .status-time {{ color: var(--text-muted); font-size: 0.7rem; }}

    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
    .dot-active   {{ background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.2s infinite; }}
    .dot-done     {{ background: var(--success); }}
    .dot-pending  {{ background: var(--border); }}
    .dot-error    {{ background: var(--danger); box-shadow: 0 0 8px var(--danger); }}

    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}

    .chat-container {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        max-height: 420px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }}
    .chat-msg {{ margin-bottom: 0.85rem; display: flex; flex-direction: column; gap: 0.2rem; }}
    .chat-label {{ font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }}
    .chat-bubble {{ display: inline-block; padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; line-height: 1.6; max-width: 90%; }}
    .user-label  {{ color: var(--accent-glow); }}
    .bot-label   {{ color: var(--accent-2); }}
    .user-bubble {{ background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }}
    .bot-bubble  {{ background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }}

    hr {{ border: none !important; border-top: 1px solid var(--border) !important; margin: 1.25rem 0 !important; }}

    .transcript-box {{
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        font-size: 0.82rem;
        line-height: 1.8;
        max-height: 300px;
        overflow-y: auto;
        color: var(--text-muted);
        white-space: pre-wrap;
        word-break: break-word;
    }}

    .history-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }}
    .history-title {{ font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: var(--text); }}
    .history-meta {{ color: var(--text-muted); font-size: 0.72rem; margin-top: 0.15rem; }}
    .stat-chip {{
        display: inline-block; background: var(--surface-2); border: 1px solid var(--border);
        border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.68rem; color: var(--text-muted);
        margin-right: 0.4rem; margin-top: 0.5rem;
    }}

    .stProgress > div > div > div {{ background: var(--accent) !important; }}
    .stSpinner > div {{ border-top-color: var(--accent) !important; }}
    [data-testid="stMarkdownContainer"] p {{ color: var(--text) !important; }}
    label {{ color: var(--text-muted) !important; font-size: 0.8rem !important; }}

    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}
    </style>
    """, unsafe_allow_html=True)

inject_css(THEMES[st.session_state.theme])

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status_class(state: str) -> str:
    return {"active": "dot-active", "done": "dot-done", "error": "dot-error"}.get(state, "dot-pending")

def step_bar_html(label: str, icon: str, state: str, elapsed=None) -> str:
    css = step_status_class(state)
    time_html = f'<span class="status-time">{elapsed:.1f}s</span>' if elapsed is not None else ""
    return f"""
    <div class="status-bar">
        <div class="status-bar-left">
            <div class="status-dot {css}"></div>
            <span>{icon} {label}</span>
        </div>
        {time_html}
    </div>"""

def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name

def guess_media_kind(src: str):
    if not src:
        return None
    ext = os.path.splitext(src)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if src.startswith("http"):
        return "video"  # covers direct links + YouTube URLs
    return None

def run_pipeline_with_live_status(source: str, language: str, placeholders: dict, labels: dict) -> dict:
    """Runs the same steps as main.py's run_pipeline(), updating sidebar
    placeholders in place as each stage starts/finishes."""

    def set_step(key, state, elapsed=None):
        st.session_state.pipeline_steps[key] = state
        icon, label = labels[key]
        placeholders[key].markdown(step_bar_html(label, icon, state, elapsed), unsafe_allow_html=True)

    def timed(key, fn, *args):
        set_step(key, "active")
        t0 = time.time()
        try:
            out = fn(*args)
        except Exception:
            set_step(key, "error")
            raise
        elapsed = time.time() - t0
        st.session_state.step_times[key] = elapsed
        set_step(key, "done", elapsed)
        return out

    chunks = timed("audio", process_input, source)
    transcript = timed("transcript", transcribe_all, chunks, language)
    title = timed("title", generate_title, transcript)
    summary = timed("summary", summarize, transcript)

    def extract_all(t):
        return (
            extract_action_items(t),
            extract_key_decisions(t),
            extract_questions(t),
        )
    action_items, decisions, questions = timed("extract", extract_all, transcript)
    rag_chain = timed("rag", build_rag_chain, transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

def generate_followups(rag_chain, question: str, answer: str):
    prompt = (
        "Based only on this Q&A about a meeting transcript, suggest exactly 3 short, "
        "specific follow-up questions the user could ask next. "
        "Return ONLY a numbered list, one question per line, nothing else.\n\n"
        f"Q: {question}\nA: {answer}"
    )
    try:
        raw = ask_question(rag_chain, prompt)
        lines = [ln.strip(" -0123456789.").strip() for ln in raw.splitlines()]
        return [ln for ln in lines if ln][:3]
    except Exception:
        return []

def session_stats(session: dict) -> dict:
    def count_lines(text):
        if not text:
            return 0
        return len([l for l in text.splitlines() if l.strip()]) or (1 if text.strip() else 0)
    return {
        "total_time": sum(session.get("step_times", {}).values()),
        "action_items": count_lines(session.get("action_items", "")),
        "decisions": count_lines(session.get("key_decisions", "")),
        "questions": count_lines(session.get("open_questions", "")),
        "words": len(session.get("transcript", "").split()),
    }

def ask_and_append(session: dict, question: str):
    with st.spinner("Thinking…"):
        answer = ask_question(session["rag_chain"], question)
    session["chat_history"].append({"role": "user", "content": question})
    session["chat_history"].append({"role": "assistant", "content": answer})
    session["followups"] = generate_followups(session["rag_chain"], question, answer)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)

    is_light = st.toggle("☀️ Light mode", value=(st.session_state.theme == "light"))
    new_theme = "light" if is_light else "dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)

    source_text = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")
    uploaded_file = st.file_uploader(
        "Or upload an audio/video file",
        type=["mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a", "ogg", "flac"],
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True, disabled=st.session_state.processing)

    st.markdown("---")
    st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)

    # Placeholders created up front so update_step() calls can push live
    # updates into them mid-run (Streamlit flushes st.empty() updates to
    # the browser immediately, without waiting for the script to finish).
    step_placeholders = {}
    labels = {key: (icon, label) for key, icon, label in STEPS}
    for key, icon, label in STEPS:
        state = st.session_state.pipeline_steps.get(key, "pending")
        elapsed = st.session_state.step_times.get(key)
        step_placeholders[key] = st.empty()
        step_placeholders[key].markdown(step_bar_html(label, icon, state, elapsed), unsafe_allow_html=True)

    if st.session_state.sessions:
        st.markdown("---")
        st.markdown('<span class="badge badge-cyan">Recent Sessions</span>', unsafe_allow_html=True)
        for i in reversed(range(len(st.session_state.sessions))):
            s = st.session_state.sessions[i]
            label = f"{'📌 ' if i == st.session_state.active_idx else ''}{s['title'][:28]}"
            if st.button(label, key=f"hist_btn_{i}", use_container_width=True):
                st.session_state.active_idx = i
                st.rerun()

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    effective_source = None
    media_bytes = None
    media_kind = None

    if uploaded_file is not None:
        effective_source = save_uploaded_file(uploaded_file)
        media_bytes = uploaded_file.getvalue()
        media_kind = "video" if uploaded_file.type.startswith("video") else "audio"
    elif source_text.strip():
        effective_source = source_text.strip()
        media_kind = guess_media_kind(effective_source)

    if not effective_source:
        st.error("Please enter a YouTube URL / file path, or upload a file.")
    else:
        st.session_state.pipeline_steps = {}
        st.session_state.step_times = {}
        st.session_state.processing = True

        for key, icon, label in STEPS:
            step_placeholders[key].markdown(step_bar_html(label, icon, "pending"), unsafe_allow_html=True)

        progress_placeholder = st.empty()
        progress_placeholder.info("⚙️ Pipeline running — watch the sidebar for live progress…")

        try:
            result = run_pipeline_with_live_status(effective_source, language, step_placeholders, labels)
            session = {
                **result,
                "source": effective_source,
                "language": language,
                "media_bytes": media_bytes,
                "media_kind": media_kind,
                "timestamp": datetime.now(),
                "step_times": dict(st.session_state.step_times),
                "chat_history": [],
                "followups": [],
            }
            st.session_state.sessions.append(session)
            st.session_state.active_idx = len(st.session_state.sessions) - 1
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.4)
            progress_placeholder.empty()
        except Exception as e:
            progress_placeholder.error(f"❌ Error: {e}")
        finally:
            st.session_state.processing = False
            st.rerun()

# ── Tabs: Analyze / Dashboard ─────────────────────────────────────────────────
tab_analyze, tab_dashboard = st.tabs(["🎬 Analyze", "📊 Dashboard"])

with tab_analyze:
    if st.session_state.active_idx is not None and st.session_state.sessions:
        r = st.session_state.sessions[st.session_state.active_idx]

        # Embedded media player
        if r.get("media_bytes") is not None:
            if r.get("media_kind") == "audio":
                st.audio(r["media_bytes"])
            else:
                st.video(r["media_bytes"])
        else:
            kind = r.get("media_kind")
            src = r.get("source", "")
            try:
                if kind == "audio":
                    st.audio(src)
                elif kind == "video":
                    st.video(src)
            except Exception:
                pass  # not all local paths / links are playable — fail silently

        st.markdown(f"""
        <div class="card">
            <div class="card-title">📌 Session Title</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
                {r['title']}
            </div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2], gap="medium")

        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📋 Summary</div>
                <div class="card-content">{r['summary']}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            with st.expander("📝 Full Transcript", expanded=False):
                st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download Transcript", r["transcript"],
                    file_name="transcript.txt", use_container_width=True,
                    key=f"dl_transcript_{st.session_state.active_idx}",
                )

        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">✅ Action Items</div>
                <div class="card-content">{r['action_items']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔑 Key Decisions</div>
                <div class="card-content">{r['key_decisions']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">❓ Open Questions</div>
                <div class="card-content">{r['open_questions']}</div>
            </div>""", unsafe_allow_html=True)

        report_md = (
            f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n"
            f"## Action Items\n{r['action_items']}\n\n"
            f"## Key Decisions\n{r['key_decisions']}\n\n"
            f"## Open Questions\n{r['open_questions']}\n\n"
            f"## Full Transcript\n{r['transcript']}\n"
        )
        st.download_button(
            "⬇ Download Full Report (.md)", report_md,
            file_name=f"{r['title'][:40] or 'report'}.md",
            key=f"dl_report_{st.session_state.active_idx}",
        )

        st.markdown("---")
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

        if r["chat_history"]:
            chat_html = '<div class="chat-container">'
            for msg in r["chat_history"]:
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-end">
                        <span class="chat-label user-label">You</span>
                        <div class="chat-bubble user-bubble">{msg['content']}</div>
                    </div>"""
                else:
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-start">
                        <span class="chat-label bot-label">🤖 Assistant</span>
                        <div class="chat-bubble bot-bubble">{msg['content']}</div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
            </div>""", unsafe_allow_html=True)

        # Suggested follow-up questions
        if r.get("followups"):
            st.markdown('<div style="font-size:0.72rem;color:var(--text-muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">Suggested follow-ups</div>', unsafe_allow_html=True)
            fcols = st.columns(len(r["followups"]))
            for i, fq in enumerate(r["followups"]):
                with fcols[i]:
                    st.markdown('<div class="followup-btn">', unsafe_allow_html=True)
                    if st.button(fq, key=f"followup_{st.session_state.active_idx}_{i}_{fq[:10]}"):
                        ask_and_append(r, fq)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        with st.form(key=f"chat_form_{st.session_state.active_idx}", clear_on_submit=True):
            chat_col1, chat_col2 = st.columns([5, 1], gap="small")
            with chat_col1:
                user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
            with chat_col2:
                send_btn = st.form_submit_button("Send →", use_container_width=True)

        if send_btn and user_input.strip():
            ask_and_append(r, user_input.strip())
            st.rerun()

        if r["chat_history"]:
            if st.button("🗑️ Clear Chat", type="secondary", key=f"clear_chat_{st.session_state.active_idx}"):
                r["chat_history"] = []
                r["followups"] = []
                st.rerun()

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
            <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
                Ready to Analyse
            </div>
            <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
                Paste a YouTube URL / file path, or upload a file, in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
            </div>
            <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
                <span class="badge badge-purple">Transcription</span>
                <span class="badge badge-cyan">Summarisation</span>
                <span class="badge badge-green">RAG Chat</span>
            </div>
        </div>""", unsafe_allow_html=True)

with tab_dashboard:
    if not st.session_state.sessions:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:var(--text-muted)">
            No sessions yet — run an analysis to see it appear here.
        </div>""", unsafe_allow_html=True)
    else:
        total = len(st.session_state.sessions)
        total_words = sum(session_stats(s)["words"] for s in st.session_state.sessions)
        total_time = sum(session_stats(s)["total_time"] for s in st.session_state.sessions)

        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="card"><div class="card-title">Total Sessions</div><div class="card-content" style="font-size:1.6rem;font-weight:700">{total}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="card"><div class="card-title">Transcript Words</div><div class="card-content" style="font-size:1.6rem;font-weight:700">{total_words:,}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="card"><div class="card-title">Total Processing Time</div><div class="card-content" style="font-size:1.6rem;font-weight:700">{total_time:.0f}s</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        for i in reversed(range(len(st.session_state.sessions))):
            s = st.session_state.sessions[i]
            stats = session_stats(s)
            ts = s["timestamp"].strftime("%b %d, %Y · %H:%M")
            colA, colB = st.columns([5, 1])
            with colA:
                st.markdown(f"""
                <div class="history-card">
                    <div class="history-title">{s['title']}</div>
                    <div class="history-meta">{ts} · {s['language']}</div>
                    <div>
                        <span class="stat-chip">⏱ {stats['total_time']:.1f}s</span>
                        <span class="stat-chip">📝 {stats['words']} words</span>
                        <span class="stat-chip">✅ {stats['action_items']} actions</span>
                        <span class="stat-chip">🔑 {stats['decisions']} decisions</span>
                        <span class="stat-chip">❓ {stats['questions']} questions</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with colB:
                st.markdown("<div style='margin-top:1.1rem'></div>", unsafe_allow_html=True)
                if st.button("Open →", key=f"open_session_{i}", use_container_width=True):
                    st.session_state.active_idx = i
                    st.rerun()