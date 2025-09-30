# server.py
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(title="Realezy Voice Support (Realtime)")

# CORS (local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"ok": True, "model": "gpt-realtime"}


def _read_text(filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            t = f.read().strip()
            return " ".join(t.split())
    except FileNotFoundError:
        return ""


@app.get("/content")
async def get_content():
    """Serve promotion and release content to the client."""
    promotion = _read_text("promotion.txt")
    release = _read_text("release.txt")
    return {"promotion": promotion, "release": release}

@app.post("/session")
async def create_session(req: Request):
    """
    Returns a short-lived client_secret for WebRTC with gpt-realtime.
    """
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "Missing OPENAI_API_KEY"})

    try:
        body = (
            await req.json()
            if req.headers.get("content-type", "").startswith("application/json")
            else {}
        )
    except Exception:
        body = {}

    voice = body.get("voice", "alloy")
    model = body.get("model", "gpt-realtime")

    promotion_text = _read_text("promotion.txt")
    release_text = _read_text("release.txt")

    def _ascii(text: str) -> str:
        try:
            return text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            return text

    promotion_ascii = _ascii(promotion_text)
    release_ascii = _ascii(release_text)

    # Session-wide English + Realezy/property scope
    session_instructions = (
        "You are a helpful voice assistant for Realezy. "
        "You must Talk only in English in Singapore accent. "
        "Follow this call flow exactly:"
        "1) Strictly maintain this Open with a polite greeting, state you are calling from Realezy, and give a short line about what Realezy does. "
        "2) Explain that you would like to share some Realezy offers and ask if the listener wants to hear them. "
        "If they decline, thank them warmly and end the call. "
        "If they accept, ask for their name and email address; if they decline either, acknowledge that's fine and continue. "
        "Deliver only the promotion content from the PROMOTION document regardless of whether they share their details, and do not reuse the PROMOTION document unless they explicitly ask to hear promotions again. "
        "3) After sharing the promotion, ask if they would like to know more about Realezy. "
        "For any questions about Realezy or its services, respond strictly using the RELEASE document and never from the PROMOTION document. "
        "If you cannot find an answer in the RELEASE document, say you have noted their question and that someone from Realezy will follow up. "
        f"PROMOTION DOCUMENT (for step 2 only):\n-----\n{promotion_ascii}\n-----\n"
        f"RELEASE DOCUMENT (authoritative for questions):\n-----\n{release_ascii}\n-----\n"
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "voice": voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "modalities": ["text", "audio"],
                    # Force English mic transcription
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "instructions": session_instructions,
                },
            )

        if r.status_code >= 400:
            return JSONResponse(
                status_code=500,
                content={"error": "Session create failed", "details": r.text},
            )

        data = r.json()
        return {
            "client_secret": data.get("client_secret", {}).get("value"),
            "model": model,
            "voice": voice,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "server_error", "details": str(e)})

# Serve static frontend
app.mount("/", StaticFiles(directory="public", html=True), name="static")

