#!/usr/bin/env python3
# main.py - rutina diaria: obtener último vídeo, transcripción, resumir (Cloudflare AI) y enviar por WhatsApp.

import os
import re
import json
import requests
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from twilio.rest import Client
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

try:
    import nltk
    nltk.download('punkt', quiet=True)
except Exception:
    pass

# === VARIABLES DE CONFIGURACIÓN ===
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "12000"))
YOUTUBER = os.getenv("YOUTUBER", "@JoseLuisCavatv")
LAST_VIDEO_FILE = os.getenv("LAST_VIDEO_FILE", "last_video.json")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "whatsapp:+34665447902")
MAX_WHATSAPP_CHARS = int(os.getenv("MAX_WHATSAPP_CHARS", "1500"))

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def safe_print(*a, **kw):
    print(*a, **kw)

def sanitize_transcript(raw_text):
    """Limpia texto o JSON devuelto por TranscriptAPI / YouTube."""
    text = raw_text if raw_text else ""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            j = json.loads(stripped)
            if isinstance(j, dict):
                for k in ("transcript", "text", "captions", "body"):
                    if k in j:
                        val = j[k]
                        if isinstance(val, list):
                            text = " ".join(str(v) for v in val)
                        elif isinstance(val, dict):
                            inner = val.get("text") or val.get("transcript")
                            text = inner if inner else json.dumps(val)
                        else:
                            text = str(val)
                        break
                else:
                    text = " ".join(str(v) for v in j.values())
            elif isinstance(j, list):
                parts = []
                for it in j:
                    if isinstance(it, dict):
                        parts.append(it.get("text") or json.dumps(it))
                    else:
                        parts.append(str(it))
                text = " ".join(parts)
        except Exception:
            pass
    # limpiar timestamps y residuos
    text = re.sub(r'\[\s*\d+[^\]]*?s\s*\]', ' ', text)
    text = re.sub(r'\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]', ' ', text)
    text = re.sub(r'\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)', ' ', text)
    text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', ' ', text)
    text = re.sub(r'\b\d+\.\s*\d+s\b', ' ', text)
    text = re.sub(r'[\{\}\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================================================
# YOUTUBE
# =========================================================
def get_latest_video_id(channel_identifier):
    if youtube is None:
        raise RuntimeError("YOUTUBE_API_KEY faltante")

    if channel_identifier.startswith("@"):
        resp = youtube.search().list(part="snippet", q=channel_identifier, type="channel", maxResults=1).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError("No se encontró canal con ese handle.")
        item = items[0]
        channel_id = item.get("id", {}).get("channelId") or item.get("snippet", {}).get("channelId")
    elif channel_identifier.startswith("UC"):
        channel_id = channel_identifier
    else:
        resp = youtube.channels().list(part="id", forUsername=channel_identifier).execute()
        channel_id = resp["items"][0]["id"]

    details = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    uploads_playlist_id = details["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    playlist_items = youtube.playlistItems().list(playlistId=uploads_playlist_id, part="snippet", maxResults=1).execute()
    snippet = playlist_items["items"][0]["snippet"]
    return snippet["resourceId"]["videoId"], snippet.get("title", "Sin título")

def is_new_video(video_id):
    if not os.path.exists(LAST_VIDEO_FILE):
        return True
    with open(LAST_VIDEO_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return True
    return data.get("last_video_id") != video_id

def save_last_video(video_id):
    with open(LAST_VIDEO_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_video_id": video_id}, f)

# =========================================================
# TRANSCRIPCIÓN
# =========================================================
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
        text = " ".join([t.get("text", "") for t in transcript])
        safe_print("✅ Transcripción obtenida desde subtítulos de YouTube.")
        return sanitize_transcript(text)
    except Exception as e:
        safe_print("⚠️ No hay subtítulos:", str(e))
        safe_print("↪ Intentando TranscriptAPI...")
        api_key = os.getenv("TRANSCRIPTAPI_KEY")
        url = "https://transcriptapi.com/api/v2/youtube/transcript"
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.get(url, headers=headers, params={"video_url": video_url, "format": "text"}, timeout=90)
        if resp.status_code != 200:
            raise ValueError(f"❌ Error TranscriptAPI: {resp.status_code}")
        safe_print("✅ Transcripción obtenida con TranscriptAPI.")
        return sanitize_transcript(resp.text)

# =========================================================
# RESUMEN CON CLOUDFLARE
# =========================================================
def summarize_with_cloudflare(text, model="@cf/meta/llama-2-7b-chat-int8", sentences=8):
    """Usa Cloudflare Workers AI para resumir texto."""
    account_id = os.getenv("CF_ACCOUNT_ID")
    api_token = os.getenv("CF_API_TOKEN")
    if not account_id or not api_token:
        raise RuntimeError("Faltan CF_ACCOUNT_ID / CF_API_TOKEN")

    sys_prompt = "Eres un asistente que resume textos en español de forma precisa y concisa."
    user_prompt = f"Resume el siguiente texto en {sentences} frases claras y estructuradas:\n\n{text[:200000]}"

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    result = data.get("result", {})
    return (result.get("response") or str(data)).strip()

# =========================================================
# ENVÍO WHATSAPP
# =========================================================
def send_whatsapp_message(text):
    """Envía el texto en partes si supera el límite de caracteres."""
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    parts = []
    if len(text) <= MAX_WHATSAPP_CHARS:
        parts = [text]
    else:
        remaining = text
        while remaining:
            if len(remaining) <= MAX_WHATSAPP_CHARS:
                parts.append(remaining)
                break
            cut = remaining.rfind("\n\n", 0, MAX_WHATSAPP_CHARS)
            if cut == -1:
                cut = remaining.rfind(".", 0, MAX_WHATSAPP_CHARS)
            if cut == -1 or cut < MAX_WHATSAPP_CHARS // 2:
                cut = MAX_WHATSAPP_CHARS
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()

    for p in parts:
        msg = client.messages.create(from_=TWILIO_WHATSAPP_FROM, to=TWILIO_WHATSAPP_TO, body=p)
        safe_print("Mensaje enviado:", msg.sid)

# =========================================================
# MAIN
# =========================================================
def main():
    safe_print("Inicio rutina: buscar último vídeo...")
    try:
        video_id, title = get_latest_video_id(YOUTUBER)
        safe_print("Último vídeo:", title, "(", video_id, ")")

        if not is_new_video(video_id):
            safe_print("No hay vídeo nuevo.")
            return

        safe_print("Obteniendo transcripción...")
        transcript = get_transcript(video_id)

        safe_print("Generando resumen con Cloudflare...")
        resumen_text = summarize_with_cloudflare(transcript)

        resumen = f"🧠 Resumen del vídeo '{title}':\n\n{resumen_text}"
        safe_print("Enviando por WhatsApp...")
        send_whatsapp_message(resumen)

        save_last_video(video_id)
        safe_print("✅ Proceso completado correctamente.")
    except Exception as e:
        safe_print("❌ Error en ejecución:", repr(e))


if __name__ == "__main__":
    main()
