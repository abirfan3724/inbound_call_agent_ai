import asyncio
import logging
import os
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("realezy.voice")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_CALLER_ID = os.getenv("TWILIO_CALLER_ID")
TWILIO_DEFAULT_TO = os.getenv("TWILIO_DEFAULT_TO")
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL")
TWILIO_TTS_VOICE = os.getenv("TWILIO_TTS_VOICE", "Polly.Joanna")
TWILIO_TTS_LANGUAGE = os.getenv("TWILIO_TTS_LANGUAGE", "en-US")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

logger.info(
    "Config loaded | webhook=%s | caller_id=%s | default_to=%s",
    TWILIO_WEBHOOK_URL,
    TWILIO_CALLER_ID,
    TWILIO_DEFAULT_TO,
)

_twilio_client: Optional[Client] = None
_call_state: Dict[str, Dict[str, object]] = {}
END_MARKER = "<<<END_CALL>>>"

app = FastAPI(title="Realezy Voice Support (Realtime)")

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
        logger.warning("Document missing: %s", filename)
        return ""


def _ascii(text: str) -> str:
    try:
        return text.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return text


def _load_documents() -> Dict[str, str]:
    promotion_text = _read_text("promotion.txt")
    release_text = _read_text("release.txt")
    logger.info(
        "Docs loaded | promotion_len=%d | release_len=%d",
        len(promotion_text),
        len(release_text),
    )
    return {
        "promotion": promotion_text,
        "release": release_text,
        "promotion_ascii": _ascii(promotion_text),
        "release_ascii": _ascii(release_text),
    }


def _compose_base_instructions(promotion_ascii: str, release_ascii: str) -> str:
    return (
        "You are a helpful voice assistant for Realezy. Your name is Jane from Realezy's sales team. "
        "If the caller asks for your name, introduce yourself as Jane from Realezy. "
        "If the caller greets you or makes small talk (for example, 'Hi' or 'How are you?'), respond warmly and politely. "
        "Speak only in English and keep your tone natural, friendly, and conversational - never robotic. "
        "Follow this call flow exactly: "
        "1) Open with a polite greeting, state you are calling from Realezy, and give a short line about what Realezy does. "
        "2) Explain that you would like to share some Realezy offers and ask if the listener wants to hear them. "
        "If they decline, thank them warmly and end the call. "
        "3) If they accept, ask for their name and email address; if they decline either, acknowledge that's fine and continue. "
        "4) Deliver only the promotion content from the PROMOTION document regardless of whether they share their details, and do not reuse the PROMOTION document unless they explicitly ask to hear the promotions again. "
        "When the caller asks follow-up questions about the promotion (such as pricing or what is included), answer directly using the PROMOTION document. "
        "5) After sharing the promotion, ask if they would like to know more about Realezy. "
        "For any other questions about Realezy or its services, respond strictly using the RELEASE document. "
        "Do not restart the call from the beginning after the initial greeting unless the caller clearly asks you to start over, even if there is background noise, a side conversation, or a short interruption. "
        "If you hear the caller speaking to someone else nearby, wait politely, acknowledge that you will hold, and resume from the point where you left off once they respond. "
        "If you hear brief ambient noise such as traffic, street sounds, or other people nearby but the caller has not asked you to repeat yourself, stay on topic and continue from where you left off without restarting. "
        "If you encounter sustained loud background noise or the caller sounds busy, acknowledge it, summarize the last point you covered, and ask if they would prefer you call back later (for example, 'If now is not a good time, I can call you back later. Would that help?'). Wait for their answer before deciding whether to continue or end the call. "
        "Only end the call after a friendly closing once all questions are addressed or the caller agrees to wrap up. "
        "If you cannot find an answer in either the PROMOTION or RELEASE document, say you have noted their question and that someone from Realezy will follow up. "
        f"PROMOTION DOCUMENT (for promotions and related questions):\n-----\n{promotion_ascii}\n-----\n"
        f"RELEASE DOCUMENT (authoritative for other questions):\n-----\n{release_ascii}\n-----\n"
    )


def _compose_voice_instructions(promotion_ascii: str, release_ascii: str) -> str:
    base = _compose_base_instructions(promotion_ascii, release_ascii)
    return (
        base
        + "Always speak in short, friendly sentences that sound natural when read aloud. "
        + "Ask one question at a time and wait for the caller to answer before moving on. "
        + "When the listener explicitly says goodbye, declines further help, or you are confident the chat is finished, append the marker <<<END_CALL>>> after your final sentence. "
        + "Do not mention the marker itself to the caller. "
        + "If you hear background noise or other voices, stay on the line, refer back to the last thing you said, and ask if they can still hear you instead of restarting the conversation. "
        + "Ignore brief ambient noises such as passing cars or people nearby unless the caller mentions difficulty hearing you, and continue speaking from the same point. "
        + "If the caller is speaking to someone else, pause, let them finish, and confirm when they are ready before continuing from where you left off. "
        + "When the noise persists or the caller sounds distracted, offer to call back later as described, and only end the call after they confirm it is a better option. "
    )


def _build_opening_message(docs: Dict[str, str]) -> str:
    promo = docs.get("promotion") or "We have a fresh promotion running right now."
    promo_snippet = promo.split(".")
    first_sentence = promo_snippet[0].strip()
    if first_sentence:
        highlight = first_sentence
    else:
        highlight = promo[:160]
    highlight = highlight.strip()
    if highlight and not highlight.endswith("."):
        highlight += "."
    return (
        "Hello! This is Realezy calling. We help Singapore home buyers with flexible property financing options. "
        f"Here''s a quick highlight: {highlight} "
        "Is this a good time to share the latest offers with you?"
    )


def _get_twilio_client() -> Optional[Client]:
    global _twilio_client
    if _twilio_client is None and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        logger.info("Creating Twilio client")
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


async def _openai_chat(messages: List[Dict[str, str]]) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.error("Missing OPENAI_API_KEY; cannot contact OpenAI")
        return None

    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if resp.status_code >= 400:
            logger.error(
                "OpenAI error | status=%s | body=%s",
                resp.status_code,
                resp.text,
            )
            return None
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            logger.info("OpenAI reply (trimmed) | %s", content[:200])
            return content.strip()
    except Exception as exc:
        logger.exception("OpenAI call failed: %s", exc)
        return None
    return None


def _twiml_response(message: str, end_call: bool = False) -> str:
    safe_message = escape(_ascii(message)) if message else ""
    logger.info("Building TwiML | end_call=%s | msg=%s", end_call, safe_message[:160])
    if end_call:
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Response>"
            f"<Say voice=\"{TWILIO_TTS_VOICE}\" language=\"{TWILIO_TTS_LANGUAGE}\">{safe_message}</Say>"
            "<Hangup/>"
            "</Response>"
        )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        f"<Say voice=\"{TWILIO_TTS_VOICE}\" language=\"{TWILIO_TTS_LANGUAGE}\">{safe_message}</Say>"
        f"<Gather input=\"speech\" action=\"/twilio/handle-input\" method=\"POST\" speechTimeout=\"auto\" language=\"{TWILIO_TTS_LANGUAGE}\"></Gather>"
        f"<Say voice=\"{TWILIO_TTS_VOICE}\" language=\"{TWILIO_TTS_LANGUAGE}\">I did not hear anything, so we will follow up later. Goodbye.</Say>"
        "<Hangup/>"
        "</Response>"
    )


def _ensure_call_state(call_sid: str) -> Dict[str, object]:
    state = _call_state.get(call_sid)
    if state is None:
        logger.info("Creating new call state | sid=%s", call_sid)
        docs = _load_documents()
        instructions = _compose_voice_instructions(
            docs["promotion_ascii"], docs["release_ascii"]
        )
        opening_message = _build_opening_message(docs)
        state = {
            "history": [
                {"role": "system", "content": instructions},
                {"role": "assistant", "content": opening_message},
            ],
            "started": False,
            "last_assistant": opening_message,
        }
        logger.info("Call %s | opening message prepared: %s", call_sid, opening_message)
        _call_state[call_sid] = state
    return state


async def _generate_voice_reply(call_sid: str, user_text: Optional[str]) -> str:
    state = _ensure_call_state(call_sid)
    history: List[Dict[str, str]] = state["history"]  # type: ignore[arg-type]

    if user_text:
        logger.info("Call %s | user said: %s", call_sid, user_text)
        history.append({"role": "user", "content": user_text})

    assistant_text = await _openai_chat(history)
    if not assistant_text:
        assistant_text = (
            "Apologies, I had trouble formulating a response just now. Could you please repeat that?"
        )
    history.append({"role": "assistant", "content": assistant_text})
    logger.info("Call %s | assistant reply: %s", call_sid, assistant_text)

    if len(history) > 15:
        trimmed = [history[0]] + history[-14:]
        state["history"] = trimmed

    state["last_assistant"] = assistant_text
    state["started"] = True
    return assistant_text


@app.get("/content")
async def get_content():
    logger.info("/content requested")
    docs = _load_documents()
    return {
        "promotion": docs["promotion"],
        "release": docs["release"],
    }


@app.post("/session")
async def create_session(req: Request):
    logger.info("/session create requested")
    if not OPENAI_API_KEY:
        logger.error("Cannot create session without OPENAI_API_KEY")
        return JSONResponse(status_code=500, content={"error": "Missing OPENAI_API_KEY"})

    try:
        body = (
            await req.json()
            if req.headers.get("content-type", "").startswith("application/json")
            else {}
        )
    except Exception:
        body = {}

    logger.info("Session body: %s", body)
    voice = body.get("voice", "Ash")
    model = body.get("model", "gpt-realtime")

    docs = _load_documents()
    session_instructions = _compose_base_instructions(
        docs["promotion_ascii"], docs["release_ascii"]
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
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "instructions": session_instructions,
                },
            )

        if r.status_code >= 400:
            logger.error("Realtime session failed | status=%s | body=%s", r.status_code, r.text)
            return JSONResponse(
                status_code=500,
                content={"error": "Session create failed", "details": r.text},
            )

        data = r.json()
        logger.info("Realtime session created")
        return {
            "client_secret": data.get("client_secret", {}).get("value"),
            "model": model,
            "voice": voice,
        }

    except Exception as e:
        logger.exception("Realtime session error: %s", e)
        return JSONResponse(status_code=500, content={"error": "server_error", "details": str(e)})


@app.post("/call")
async def initiate_call(req: Request):
    logger.info("/call invoked")
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_CALLER_ID):
        logger.error("Missing Twilio configuration")
        return JSONResponse(status_code=500, content={"error": "missing_twilio_config"})

    try:
        body = (
            await req.json()
            if req.headers.get("content-type", "").startswith("application/json")
            else {}
        )
    except Exception:
        body = {}

    logger.info("Call request body: %s", body)
    to_number = body.get("to") or TWILIO_DEFAULT_TO
    if not to_number:
        logger.error("Destination number missing")
        return JSONResponse(status_code=400, content={"error": "missing_destination"})

    docs = _load_documents()
    message = body.get("message")
    if not message:
        message = (
            "Hello! This is Realezy calling to share our latest offers. "
            f"{docs['promotion'] or 'Visit realezy.com for more details.'}"
        )

    voice = body.get("voice", TWILIO_TTS_VOICE)
    language = body.get("language", TWILIO_TTS_LANGUAGE)

    twiml = body.get("twiml")
    if not twiml:
        safe_message = escape(_ascii(message))
        twiml = (
            f'<Response><Say voice="{voice}" language="{language}">' f"{safe_message}</Say></Response>"
        )

    client = _get_twilio_client()
    if client is None:
        logger.error("Twilio client unavailable")
        return JSONResponse(status_code=500, content={"error": "twilio_client_unavailable"})

    call_kwargs = {
        "to": to_number,
        "from_": TWILIO_CALLER_ID,
    }

    webhook = body.get("webhook_url") or TWILIO_WEBHOOK_URL
    if webhook:
        base_webhook = webhook.rstrip("/")
        call_kwargs["url"] = base_webhook + "/twilio/voice"
        call_kwargs["status_callback"] = base_webhook + "/twilio/status"
        call_kwargs["status_callback_event"] = [
            "initiated",
            "ringing",
            "answered",
            "completed",
            "busy",
            "failed",
            "no-answer",
        ]
        call_kwargs["status_callback_method"] = "POST"
        logger.info("Using webhook URL for call: %s", call_kwargs["url"])
    else:
        call_kwargs["twiml"] = twiml
        logger.info("Using inline TwiML for call")

    try:
        call = await asyncio.to_thread(
            client.calls.create,
            **call_kwargs,
        )
        logger.info(
            "Twilio call created | sid=%s | to=%s | status=%s",
            call.sid,
            to_number,
            call.status,
        )
        return {
            "sid": call.sid,
            "status": call.status,
            "to": to_number,
            "using_webhook": bool(webhook),
        }
    except TwilioRestException as exc:
        logger.exception("Twilio REST error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": "twilio_error", "details": str(exc)},
        )
    except Exception as exc:
        logger.exception("Call create error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "call_error", "details": str(exc)},
        )


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    logger.info("/twilio/voice hit | method=%s", request.method)
    data: Dict[str, str]
    if request.method == "POST":
        try:
            data = dict(await request.form())
        except Exception:
            logger.exception("Unable to parse POST form")
            data = {}
    else:
        data = dict(request.query_params)

    logger.info("/twilio/voice payload: %s", data)

    call_sid = data.get("CallSid")
    if not call_sid:
        xml = _twiml_response(
            "We were unable to identify the call session. Goodbye.", end_call=True
        )
        return Response(content=xml, media_type="application/xml")

    state = _ensure_call_state(call_sid)
    assistant_text = state.get("last_assistant")  # type: ignore[assignment]

    if not assistant_text:
        assistant_text = await _generate_voice_reply(
            call_sid,
            "The outbound call has connected. Greet the listener and follow the call flow instructions from the top.",
        )

    end_call = False
    if END_MARKER in assistant_text:
        assistant_text = assistant_text.replace(END_MARKER, "").strip()
        end_call = True
        _call_state.pop(call_sid, None)
        logger.info("Call %s ending by assistant instruction", call_sid)

    xml = _twiml_response(assistant_text, end_call=end_call)
    state["last_assistant"] = None
    return Response(content=xml, media_type="application/xml")


@app.post("/twilio/handle-input")
async def twilio_handle_input(request: Request):
    logger.info("/twilio/handle-input hit")
    try:
        data = dict(await request.form())
    except Exception:
        logger.exception("Unable to parse gather form")
        data = {}

    logger.info("Gather payload: %s", data)

    call_sid = data.get("CallSid")
    speech_result = (data.get("SpeechResult") or "").strip()

    if not call_sid:
        logger.error("Gather missing CallSid")
        xml = _twiml_response("We lost the call context. Goodbye.", end_call=True)
        return Response(content=xml, media_type="application/xml")

    if not speech_result:
        logger.info("Call %s | no speech detected", call_sid)
        reply = "I did not catch that. Could you please repeat?"
        xml = _twiml_response(reply)
        return Response(content=xml, media_type="application/xml")

    assistant_text = await _generate_voice_reply(call_sid, speech_result)

    end_call = False
    if END_MARKER in assistant_text:
        assistant_text = assistant_text.replace(END_MARKER, "").strip()
        end_call = True
        _call_state.pop(call_sid, None)
        logger.info("Call %s ending after gather", call_sid)

    xml = _twiml_response(assistant_text, end_call=end_call)
    state = _call_state.get(call_sid)
    if state:
        state["last_assistant"] = None
    return Response(content=xml, media_type="application/xml")


@app.post("/twilio/status")
async def twilio_status(request: Request):
    logger.info("/twilio/status hit")
    try:
        data = dict(await request.form())
    except Exception:
        logger.exception("Unable to parse status callback")
        data = {}
    logger.info("Status callback payload: %s", data)
    return Response(content="OK", media_type="text/plain")


app.mount("/", StaticFiles(directory="public", html=True), name="static")
