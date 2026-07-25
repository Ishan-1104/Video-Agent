# 🎬 AI Video Assistant

**Turn any meeting recording or YouTube video into a searchable, chattable knowledge base.**

AI Video Assistant transcribes audio/video, generates a title and summary, extracts action items, key decisions, and open questions, and lets you **chat with the transcript** using Retrieval-Augmented Generation (RAG) — all through a CLI or a polished Streamlit web UI.

---

## ✨ Features

- 🔊 **Flexible input** — analyze a YouTube URL, a local file path, or upload an audio/video file directly in the UI
- 📝 **Automatic transcription** — supports English and Hinglish
- 🏷️ **Auto-generated title** for every session
- 📋 **AI-generated summary** of the full meeting/video
- ✅ **Action items** extraction
- 🔑 **Key decisions** extraction
- ❓ **Open questions** extraction
- 💬 **Chat with your meeting** via a RAG pipeline (Mistral AI + Chroma vector store)
- 🧠 **Suggested follow-up questions** after every chat answer
- 🕒 **Live pipeline status** — watch each processing stage (audio → transcript → title → summary → extraction → RAG) complete in real time, with per-step timing
- 🗂️ **Session history** — every analysis is kept in-session so you can revisit past transcripts and chats without re-running the pipeline
- 📊 **Multi-session dashboard** — aggregate stats across all your analyzed sessions
- 🎥 **Embedded media player** for uploaded/linked audio & video
- 🌗 **Dark / light theme toggle**
- ⬇️ **Export** transcripts and full reports (Markdown)
- 🖥️ **Two interfaces** — a scriptable CLI (`main.py`) and a full web UI (`app.py`)

## 🏗️ Architecture

```
                ┌─────────────────────┐
Input ────────► │  Audio Processor     │  YouTube URL / file path / upload
 (URL/file)     │  utils/audio_        │  → downloads & chunks audio
                │  processor.py        │
                └──────────┬───────────┘
                           │ audio chunks
                           ▼
                ┌─────────────────────┐
                │  Transcriber         │  Speech → text
                │  core/transcriber.py │  (English / Hinglish)
                └──────────┬───────────┘
                           │ transcript
             ┌─────────────┼──────────────┬───────────────┐
             ▼             ▼              ▼               ▼
      ┌────────────┐┌────────────┐┌───────────────┐┌───────────────┐
      │  Title      ││  Summary   ││  Extraction    ││  Vector Store  │
      │  generate_  ││  summarize ││  action items, ││  build_vector_ │
      │  title()    ││            ││  decisions,    ││  store()       │
      │             ││            ││  questions     ││  (Chroma +     │
      │  core/      ││  core/     ││                ││  HF embeddings)│
      │  summarize  ││  summarize ││  core/extract  ││  core/         │
      │  .py        ││  .py       ││  .py           ││  vector_store  │
      └────────────┘└────────────┘└───────────────┘└───────┬────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────────┐
                                                    │  RAG Chain            │
                                                    │  build_rag_chain()    │
                                                    │  ask_question()       │
                                                    │  core/rag_engine.py   │
                                                    │  (LangChain + Mistral)│
                                                    └──────────┬────────────┘
                                                               │
                                                               ▼
                                                     💬 Chat with your meeting
```

Each analysis session builds its **own isolated Chroma collection**, so transcripts from different sessions never mix during retrieval.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM (summary / extraction / chat) | [Mistral AI](https://mistral.ai/) via `langchain-mistralai` (`mistral-small-latest`) |
| Orchestration | [LangChain](https://www.langchain.com/) (LCEL pipelines) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Vector Store | [Chroma](https://www.trychroma.com/) via `langchain-chroma` |
| Text Splitting | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Config | `python-dotenv` |

> Update the audio/transcription rows above with the actual libraries used in `utils/audio_processor.py` and `core/transcriber.py` (e.g. `yt-dlp`, `whisper`/`faster-whisper`, `pydub`, etc.).



## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` yet, a good starting point based on the modules used in this project:
>
> ```
> streamlit
> python-dotenv
> langchain
> langchain-core
> langchain-mistralai
> langchain-huggingface
> langchain-chroma
> langchain-text-splitters
> chromadb
> sentence-transformers
> ```
>
> Plus whatever audio download/transcription libraries `audio_processor.py` and `transcriber.py` depend on (e.g. `yt-dlp`, `openai-whisper` / `faster-whisper`, `pydub`, `ffmpeg`).
>
> **Note:** `ffmpeg` must also be installed on your system and available on `PATH` for most audio pipelines.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Add any additional keys your `audio_processor`/`transcriber` modules require (e.g. a transcription API key), if applicable.

---

## 🚀 Usage

### Option A — Command Line

```bash
python main.py
```

You'll be prompted for:
- A YouTube URL or local file path
- Language (`english` or `hinglish`)

The CLI prints the title, summary, action items, key decisions, and open questions, then drops you into an interactive chat loop:

```
💬 Chat with your meeting (type 'exit' to quit)

You: What were the main decisions made?
🤖 Assistant: ...
```

### Option B — Web UI

```bash
streamlit run app.py
```

Then, in the sidebar:
1. Paste a YouTube URL / file path, **or** upload an audio/video file
2. Choose a language
3. Click **⚡ Analyse**
4. Watch each pipeline stage complete live in the sidebar
5. Explore the summary, action items, decisions, and questions
6. Chat with the transcript, click suggested follow-ups, and browse past sessions in the **Dashboard** tab

---

## 🔍 How It Works

1. **`process_input(source)`** — resolves a YouTube URL, local path, or uploaded file into audio chunks.
2. **`transcribe_all(chunks, language)`** — transcribes each chunk and stitches together the full transcript.
3. **`generate_title(transcript)`** / **`summarize(transcript)`** — LLM calls that produce a concise title and summary.
4. **`extract_action_items` / `extract_key_decisions` / `extract_questions`** — targeted LLM extraction passes over the transcript.
5. **`build_vector_store(transcript)`** — splits the transcript into ~500-character chunks (50-char overlap), embeds them with `all-MiniLM-L6-v2`, and stores them in a **session-unique Chroma collection**.
6. **`build_rag_chain(transcript)`** — wires up a retriever (top-`k=4` similarity search) into a LangChain LCEL pipeline: `retriever → format_docs → prompt → Mistral LLM → output parser`.
7. **`ask_question(rag_chain, question)`** — runs a question through the chain; the LLM answers strictly from retrieved transcript context, and says so explicitly if the answer isn't in the transcript.

---

## 🗺️ Roadmap

- [ ] Speaker diarization in the transcript
- [ ] Persistent, database-backed session history (currently in-memory per browser session)
- [ ] Multi-language summary/extraction output
- [ ] Export to Notion / Slack / email
- [ ] Authentication for multi-user deployments
- [ ] Automatic cleanup of old Chroma collections in `vector_db/`

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — update this section if you're using a different license.

---

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the LLM
- [LangChain](https://www.langchain.com/) for orchestration
- [Chroma](https://www.trychroma.com/) for vector storage
- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [Streamlit](https://streamlit.io/) for the web UI
