import asyncio
import base64
import contextlib
import json
import logging
import os
import urllib.parse
from typing import Any, Dict, Optional
from xml.sax.saxutils import escape

import audioop
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
from websockets.exceptions import ConnectionClosed

try:
    # websockets v11/12 places the asyncio client under websockets.client
    from websockets.client import connect as ws_connect
except ImportError:  # pragma: no cover - fallback for newer layouts
    from websockets.asyncio.client import connect as ws_connect

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("realezy.voice")

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_CALLER_ID = os.getenv("TWILIO_CALLER_ID")
TWILIO_DEFAULT_TO = os.getenv("TWILIO_DEFAULT_TO")
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL")
TWILIO_STREAM_URL = os.getenv("TWILIO_STREAM_URL")
OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL", "gpt-realtime"
)
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
REALTIME_SAMPLE_RATE = int(os.getenv("REALTIME_SAMPLE_RATE", "16000"))
TWILIO_AUDIO_SAMPLE_RATE = 8000
SILENCE_RMS_THRESHOLD = int(os.getenv("SILENCE_RMS_THRESHOLD", "200"))
SILENCE_FRAMES_BEFORE_COMMIT = int(os.getenv("SILENCE_FRAMES_BEFORE_COMMIT", "4"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

logger.info(
    "Config loaded | webhook=%s | stream=%s | caller_id=%s | default_to=%s",
    TWILIO_WEBHOOK_URL,
    TWILIO_STREAM_URL,
    TWILIO_CALLER_ID,
    TWILIO_DEFAULT_TO,
)

_twilio_client: Optional[Client] = None

app = FastAPI(title="Realezy Voice Support (Realtime)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "model": OPENAI_REALTIME_MODEL}


def _read_text(filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return " ".join(f.read().strip().split())
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
        "IMPORTANT: You must speak ONLY in English. "
        "If the caller uses any other language, politely ask to continue in English and keep responding in English only. "
        "You are a helpful voice assistant for Realezy. Your name is Jane from Realezy's sales team. "
        "If the caller asks for your name, introduce yourself as Jane from Realezy. "
        "If the caller greets you or makes small talk (for example, 'Hi' or 'How are you?'), respond warmly and politely. "
        "Speak only in English and keep your tone natural, friendly, and conversational - never robotic. "
        "Follow this call flow exactly: "
        "1) Open with a polite greeting, state you are calling from Realezy, and give a short line about what Realezy does. "
        "2) Explain that you would like to share some Realezy offers and ask if the listener wants to hear them. "
        "If they decline, thank them warmly and end the call. "
        "3) If they accept, ask for their name and email address, wait for the caller to provide their details, and if they decline either, acknowledge that's fine and continue. "
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
        + "CRITICAL: You MUST speak in English at ALL times during this call. Never switch to any other language. "
        + "Always greet the user in English. "
        + "If the caller speaks another language, stay in English and ask them to continue in English. "
        + "Always speak in short, friendly sentences that sound natural when read aloud. "
        + "Ask one question at a time and wait for the caller to answer before moving on. "
        + "When the listener explicitly says goodbye, declines further help, or you are confident the chat is finished, wrap up with a friendly closing and let them know you will end the call. "
        + "If you hear background noise or other voices, stay on the line, refer back to the last thing you said, and ask if they can still hear you instead of restarting the conversation. "
        + "Ignore brief ambient noises such as passing cars or people nearby unless the caller mentions difficulty hearing you, and continue speaking from the same point. "
        + "If the caller is speaking to someone else, pause, let them finish, and confirm when they are ready before continuing from where you left off. "
        + "When the noise persists or the caller sounds distracted, offer to call back later as described, and only end the call after they confirm it is a better option. "
    )


def _build_stream_url(call_sid: str) -> Optional[str]:
    base = TWILIO_STREAM_URL or TWILIO_WEBHOOK_URL
    if not base:
        logger.error("Missing TWILIO_STREAM_URL/TWILIO_WEBHOOK_URL; cannot build stream URL")
        return None

    if base.startswith("http://"):
        stream_base = "ws://" + base[len("http://") :]
    elif base.startswith("https://"):
        stream_base = "wss://" + base[len("https://") :]
    else:
        stream_base = base

    stream_base = stream_base.rstrip("/")
    if not stream_base.endswith("/twilio/stream"):
        stream_base += "/twilio/stream"

    query = urllib.parse.urlencode({"call_sid": call_sid})
    return f"{stream_base}?{query}"


def _twilio_payload_to_pcm(payload: str) -> tuple[Optional[bytes], float]:
    if not payload:
        return None, 0.0
    try:
        mulaw_audio = base64.b64decode(payload)
        pcm8k = audioop.ulaw2lin(mulaw_audio, 2)
        energy = float(audioop.rms(pcm8k, 2))
        if REALTIME_SAMPLE_RATE != TWILIO_AUDIO_SAMPLE_RATE:
            pcm16k, _ = audioop.ratecv(
                pcm8k,
                2,
                1,
                TWILIO_AUDIO_SAMPLE_RATE,
                REALTIME_SAMPLE_RATE,
                None,
            )
        else:
            pcm16k = pcm8k
        return pcm16k, energy
    except Exception:
        logger.exception("Failed to decode Twilio media payload")
        return None, 0.0


def _pcm_to_twilio_payload(pcm_bytes: bytes, sample_rate: int) -> Optional[str]:
    if not pcm_bytes:
        return None
    try:
        if sample_rate != TWILIO_AUDIO_SAMPLE_RATE:
            pcm8k, _ = audioop.ratecv(
                pcm_bytes,
                2,
                1,
                sample_rate,
                TWILIO_AUDIO_SAMPLE_RATE,
                None,
            )
        else:
            pcm8k = pcm_bytes
        mulaw = audioop.lin2ulaw(pcm8k, 2)
        return base64.b64encode(mulaw).decode("ascii")
    except Exception:
        logger.exception("Failed to encode audio for Twilio")
        return None


def _twiml_response(message: str, end_call: bool = False) -> str:
    safe_excerpt = (message or "").strip()[:160]
    logger.info("Building fallback TwiML | end_call=%s | msg=%s", end_call, safe_excerpt)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Pause length=\"1\"/>"
        "<Hangup/>"
        "</Response>"
    )


def _twiml_stream(stream_url: str, call_sid: str) -> str:
    safe_url = escape(stream_url)
    safe_call_sid = escape(call_sid)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{safe_url}" bidirectional="true">'
        f'<Parameter name="callSid" value="{safe_call_sid}"/>'
        "</Stream>"
        "</Connect>"
        "</Response>"
    )


def _get_twilio_client() -> Optional[Client]:
    global _twilio_client
    if _twilio_client is None and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        logger.info("Creating Twilio client")
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


@app.get("/content")
async def get_content() -> Dict[str, str]:
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
    voice = body.get("voice", OPENAI_REALTIME_VOICE)
    model = body.get("model", OPENAI_REALTIME_MODEL)

    docs = _load_documents()
    session_instructions = _compose_voice_instructions(
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
        missing = []
        if not TWILIO_ACCOUNT_SID: missing.append("TWILIO_ACCOUNT_SID")
        if not TWILIO_AUTH_TOKEN: missing.append("TWILIO_AUTH_TOKEN")
        if not TWILIO_CALLER_ID: missing.append("TWILIO_CALLER_ID")
        logger.error("Missing Twilio configuration: %s", ", ".join(missing))
        return JSONResponse(status_code=500, content={"error": "missing_twilio_config", "details": f"Missing: {', '.join(missing)}"})

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

    client = _get_twilio_client()
    if client is None:
        logger.error("Twilio client unavailable")
        return JSONResponse(status_code=500, content={"error": "twilio_client_unavailable"})

    webhook = body.get("webhook_url") or TWILIO_WEBHOOK_URL
    if not webhook:
        logger.error("Webhook URL missing; realtime streaming requires webhook")
        return JSONResponse(status_code=400, content={"error": "missing_webhook"})

    base_webhook = webhook.rstrip("/")
    call_kwargs = {
        "to": to_number,
        "from_": TWILIO_CALLER_ID,
        "url": base_webhook + "/twilio/voice",
        "status_callback": base_webhook + "/twilio/status",
        "status_callback_method": "POST",
        "status_callback_event": [
            "initiated",
            "ringing",
            "answered",
            "completed",
            "busy",
            "failed",
            "no-answer",
        ],
    }
    logger.info("Using webhook URL for call: %s", call_kwargs["url"])

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
            "using_webhook": True,
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


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    call_sid = websocket.query_params.get("call_sid")
    try:
        await websocket.accept(subprotocol="audio.srt")
        logger.info("Twilio websocket accepted | call_sid=%s", call_sid)
    except Exception:
        logger.exception("Unable to accept Twilio websocket | call_sid=%s", call_sid)
        return

    if not OPENAI_API_KEY:
        logger.error("Missing OPENAI_API_KEY; closing Twilio stream")
        await websocket.close(code=1011, reason="missing-openai-key")
        return

    docs = _load_documents()
    instructions = _compose_voice_instructions(
        docs["promotion_ascii"], docs["release_ascii"]
    )
    logger.debug("Streaming instructions prepared | length=%d", len(instructions))

    realtime_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    realtime_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    try:
        try:
            openai_ws = await ws_connect(
                realtime_url,
                additional_headers=realtime_headers,
            )
        except TypeError:
            openai_ws = await ws_connect(
                realtime_url,
                extra_headers=realtime_headers,
            )
        logger.info("OpenAI realtime websocket connected | call_sid=%s", call_sid)
    except Exception as exc:
        logger.exception("Failed to open realtime session: %s", exc)
        await websocket.close(code=1011, reason="openai-connection-failed")
        return

    session_update = {
        "type": "session.update",
        "session": {
            "voice": OPENAI_REALTIME_VOICE,
            "instructions": instructions,
            "modalities": ["audio"],
            "input_audio_format": {
                "type": "pcm16",
                "sample_rate": REALTIME_SAMPLE_RATE,
            },
            "output_audio_format": {
                "type": "pcm16",
                "sample_rate": REALTIME_SAMPLE_RATE,
            },
            "input_audio_transcription": {
                "model": "whisper-1",
                "language": "en",
            },
        },
    }

    bridge_state: Dict[str, Any] = {
        "stream_sid": None,
        "has_audio": False,
        "silence_frames": 0,
        "awaiting_response": False,
    }
    commit_lock = asyncio.Lock()

    try:
        await openai_ws.send(json.dumps(session_update))
        # Anchor English-only + content rules as a system message in the conversation
        await openai_ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": instructions}],
                    },
                }
            )
        )
        await openai_ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "instructions": (
                            "Greet the user in English. Say exactly: 'Hello, this is Jane from Realezy.' "
                            "Always stay in English."
                        ),
                    },
                }
            )
        )
        bridge_state["awaiting_response"] = True
        logger.debug("Realtime session primed | call_sid=%s", call_sid)
    except Exception:
        logger.exception("Failed to prime realtime session")
        await openai_ws.close()
        await websocket.close(code=1011, reason="openai-session-init-failed")
        return

    async def maybe_submit_audio(force: bool = False) -> None:
        if not bridge_state["has_audio"]:
            return
        if bridge_state["awaiting_response"] and not force:
            return
        if not force and bridge_state["silence_frames"] < SILENCE_FRAMES_BEFORE_COMMIT:
            return
        async with commit_lock:
            if not bridge_state["has_audio"]:
                return
            if bridge_state["awaiting_response"] and not force:
                return
            if not force and bridge_state["silence_frames"] < SILENCE_FRAMES_BEFORE_COMMIT:
                return
            try:
                await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await openai_ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "conversation": "default",
                                "modalities": ["audio"],
                                "instructions": "Respond in English only. Keep replies brief and friendly.",
                            },
                        }
                    )
                )
                logger.debug(
                    "Committed audio buffer to OpenAI | call_sid=%s | force=%s",
                    call_sid,
                    force,
                )
                bridge_state["has_audio"] = False
                bridge_state["silence_frames"] = 0
                bridge_state["awaiting_response"] = True
            except Exception:
                logger.exception("Submitting audio to OpenAI failed")

    async def consume_twilio() -> None:
        try:
            while True:
                message = await websocket.receive_text()
                event = json.loads(message)
                event_type = event.get("event")
                if event_type == "start":
                    start_info = event.get("start", {})
                    bridge_state["stream_sid"] = start_info.get("streamSid")
                    logger.info(
                        "Twilio stream started | call_sid=%s | stream_sid=%s",
                        call_sid,
                        bridge_state["stream_sid"],
                    )
                elif event_type == "media":
                    payload = event.get("media", {}).get("payload")
                    pcm_bytes, energy = _twilio_payload_to_pcm(payload)
                    if pcm_bytes:
                        try:
                            await openai_ws.send(
                                json.dumps(
                                    {
                                        "type": "input_audio_buffer.append",
                                        "audio": base64.b64encode(pcm_bytes).decode(
                                            "ascii"
                                        ),
                                    }
                                )
                            )
                            bridge_state["has_audio"] = True
                            if energy < SILENCE_RMS_THRESHOLD:
                                bridge_state["silence_frames"] = int(
                                    bridge_state["silence_frames"]
                                ) + 1
                            else:
                                bridge_state["silence_frames"] = 0
                            await maybe_submit_audio()
                            logger.debug(
                                "Forwarded audio chunk | call_sid=%s | energy=%.2f | silence_frames=%s",
                                call_sid,
                                energy,
                                bridge_state["silence_frames"],
                            )
                        except Exception:
                            logger.exception("Failed forwarding audio chunk to OpenAI")
                elif event_type == "stop":
                    logger.info("Twilio stream stopped | call_sid=%s", call_sid)
                    break
        except WebSocketDisconnect:
            logger.info("Twilio websocket disconnected | call_sid=%s", call_sid)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Twilio websocket error | call_sid=%s", call_sid)
        finally:
            await maybe_submit_audio(force=True)

    async def consume_openai() -> None:
        try:
            async for message in openai_ws:
                event = json.loads(message)
                event_type = event.get("type")
                logger.info("OpenAI event received | type=%s | event=%s", event_type, json.dumps(event)[:200])
                if event_type == "response.audio.delta":
                    delta = event.get("delta")
                    if not delta:
                        continue
                    try:
                        pcm_bytes = base64.b64decode(delta)
                        payload = _pcm_to_twilio_payload(
                            pcm_bytes, REALTIME_SAMPLE_RATE
                        )
                        if payload and bridge_state["stream_sid"]:
                            await websocket.send_json(
                                {
                                    "event": "media",
                                    "streamSid": bridge_state["stream_sid"],
                                    "media": {"payload": payload},
                                }
                            )
                            logger.debug(
                                "Streaming audio delta back to Twilio | call_sid=%s | bytes=%d",
                                call_sid,
                                len(pcm_bytes),
                            )
                    except Exception:
                        logger.exception("Failed to send audio delta to Twilio")
                elif event_type == "response.completed":
                    bridge_state["awaiting_response"] = False
                    await maybe_submit_audio()
                    logger.debug("Realtime response completed | call_sid=%s", call_sid)
                elif event_type == "response.error":
                    bridge_state["awaiting_response"] = False
                    logger.error(
                        "Realtime response error | call_sid=%s | %s", call_sid, event
                    )
                    await maybe_submit_audio()
        except ConnectionClosed:
            logger.info("OpenAI realtime closed | call_sid=%s", call_sid)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("OpenAI realtime failure | call_sid=%s", call_sid)

    tasks = [
        asyncio.create_task(consume_twilio(), name="twilio_consumer"),
        asyncio.create_task(consume_openai(), name="openai_consumer"),
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        with contextlib.suppress(Exception):
            task.result()

    logger.info("Streaming bridge closed | call_sid=%s", call_sid)

    with contextlib.suppress(Exception):
        await openai_ws.close()
    with contextlib.suppress(Exception):
        await websocket.close()


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

    stream_url = _build_stream_url(call_sid)
    if not stream_url:
        xml = _twiml_response(
            "We are unable to start the realtime audio stream at the moment. Goodbye.",
            end_call=True,
        )
        return Response(content=xml, media_type="application/xml")

    xml = _twiml_stream(stream_url, call_sid)
    logger.info("Responding with stream TwiML | call_sid=%s | stream=%s", call_sid, stream_url)
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
