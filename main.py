#!/usr/bin/env python3
# main.py - rutina diaria: obtener último vídeo, transcripción, resumir localmente y enviar por WhatsApp.
# Requiere (si usas abstractive/hybrid): transformers, torch, sentencepiece
# Requiere siempre: google-api-python-client, youtube-transcript-api, requests, twilio, sumy

import os
import json
import math
import requests
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from twilio.rest import Client

# IMPORTS opcionales (lazy) para resumidores pesados
# from transformers import pipeline  <-- cargado solo si hace falta
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ================= CONFIGURACIÓN =================

# --- Añade esto en la parte superior, tras los imports ---
# Asegurar recursos NLTK necesarios para sumy/tokenizers (evita fallos en CI)
try:
    import nltk
    # descarga silenciosa; si ya existe no vuelve a bajar
    nltk.download('punkt', quiet=True)
except Exception as _e:
    safe_print("⚠️ No se pudo descargar recursos NLTK (punkt). Si ejecutas en CI, añade 'python -c \"import nltk; nltk.download(\"punkt\")\"' al workflow). Error:", repr(_e))

# Fallback sencillo si sumy/transformers fallan: tomar las primeras frases
def simple_fallback_summary(text, max_sentences=5, max_chars=1000):
    if not text:
        return ""
    # separar por puntos (heurística básica, muy rápida)
    sentences = [s.strip() for s in text.replace("\r","").split('.') if s.strip()]
    if not sentences:
        return text[:max_chars]
    chosen = sentences[:max_sentences]
    out = '. '.join(chosen)
    if len(out) > max_chars:
        return out[:max_chars-3].rstrip() + "..."
    return out + ('.' if not out.endswith('.') else '')

# Puedes cambiar mediante env vars:
# SUMMARIZER_METHOD: "hybrid" (default), "abstractive", "extractive"
SUMMARIZER_METHOD = os.getenv("SUMMARIZER_METHOD", "hybrid").lower()
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
EXTRACTIVE_SENTENCES = int(os.getenv("EXTRACTIVE_SENTENCES", "10"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "12000"))  # para chunking de transcripción
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "spanish")  # tokenizer language for sumy
MAX_WHATSAPP_CHARS = int(os.getenv("MAX_WHATSAPP_CHARS", "1500"))  # divide mensajes si hace falta

# YouTuber target (handle, channel id UC..., or legacy username)
YOUTUBER = os.getenv("YOUTUBER", "@JoseLuisCavatv")

# Archivo para guardar último vídeo procesado
LAST_VIDEO_FILE = os.getenv("LAST_VIDEO_FILE", "last_video.json")

# Inicializar cliente YouTube (requiere YOUTUBE_API_KEY en secretos)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    print("⚠️ ATENCIÓN: no se ha configurado YOUTUBE_API_KEY. Algunas funciones fallarán.")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

# Twilio (se usan en la función de envío)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "whatsapp:+34665447902")  # cámbialo por el tuyo

# Lazy pipeline (no crear al importar)
_transformer_pipeline = None

# ================= UTILIDADES Y HELPERS =================

def safe_print(*args, **kwargs):
    """Wrapper para prints (puedes redirigir o ampliar si quieres)."""
    print(*args, **kwargs)

# ---------- Obtener último vídeo ----------
def get_latest_video_id(channel_identifier):
    """
    Devuelve (video_id, title) del último video subido.
    Acepta: handle (@...), channel ID (UC...), o username antiguo.
    """
    if youtube is None:
        raise RuntimeError("YouTube client no inicializado (falta YOUTUBE_API_KEY).")

    # si handle
    if channel_identifier.startswith("@"):
        resp = youtube.search().list(part="snippet", q=channel_identifier, type="channel", maxResults=1).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError("No se encontró canal con ese handle.")
        item = items[0]
        # id.channelId es el campo estándar; algunos SDKs/versions pueden tenerlo en snippet.amos comprobar ambos
        channel_id = item.get("id", {}).get("channelId") or item.get("snippet", {}).get("channelId")
        if not channel_id:
            # último recurso: intentar buscar por título y obtener channelId desde snippet.channelId o buscar por nombre
            channel_id = item.get("snippet", {}).get("channelId")
        if not channel_id:
            raise ValueError("No se pudo extraer channelId del resultado de búsqueda.")
    elif channel_identifier.startswith("UC"):
        channel_id = channel_identifier
    else:
        # username antiguo
        resp = youtube.channels().list(part="id", forUsername=channel_identifier).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError("No se encontró canal con ese nombre de usuario.")
        channel_id = items[0]["id"]

    # obtener playlist de uploads
    details = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    items = details.get("items", [])
    if not items:
        raise ValueError("No se pudo obtener contentDetails del canal.")
    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    playlist_items = youtube.playlistItems().list(playlistId=uploads_playlist_id, part="snippet", maxResults=1).execute()
    if not playlist_items.get("items"):
        raise ValueError("No hay vídeos en la playlist de subidas.")
    snippet = playlist_items["items"][0]["snippet"]
    video_id = snippet["resourceId"]["videoId"]
    title = snippet.get("title", "Sin título")
    return video_id, title

# ---------- Check nuevo vídeo ----------
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

# ================= TRANSCRIPCIÓN =================

def get_transcript(video_id):
    """
    Intenta subtítulos oficiales; si no, usa transcriptapi.com con la URL completa.
    Devuelve string.
    """
    try:
        # youtube_transcript_api acepta video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
        text = " ".join([t.get("text", "") for t in transcript]).strip()
        safe_print("✅ Transcripción obtenida desde subtítulos de YouTube.")
        return text
    except Exception as e:
        safe_print("⚠️ No hay subtítulos accesibles con youtube_transcript_api:", str(e))
        safe_print("↪ Intentando TranscriptAPI (externa)...")
        api_key = os.getenv("TRANSCRIPTAPI_KEY")
        url = "https://transcriptapi.com/api/v2/youtube/transcript"
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        params = {"video_url": video_url, "format": "text"}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.get(url, headers=headers, params=params, timeout=90)
        if resp.status_code != 200:
            raise ValueError(f"❌ Error TranscriptAPI: {resp.status_code} {resp.text}")
        safe_print("✅ Transcripción obtenida con TranscriptAPI.")
        return resp.text

# ================= SUMMARIZADORES =================

def chunk_text_by_chars(text, max_chars=CHUNK_MAX_CHARS):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        if end < L:
            # intenta cortar en final de frase
            nxt = text.rfind(".", start, end)
            if nxt != -1 and nxt > start:
                end = nxt + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# Extractive - Sumy TextRank
def summarize_with_sumy(text, sentences_count=EXTRACTIVE_SENTENCES, language=TRANSCRIPT_LANGUAGE):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count)
    return " ".join([str(s) for s in summary_sentences])

# Abstractive - HuggingFace transformers (lazy init)
def _init_transformer_pipeline_if_needed():
    global _transformer_pipeline
    if _transformer_pipeline is None:
        try:
            from transformers import pipeline  # lazy import
            # device=-1 -> CPU
            _transformer_pipeline = pipeline("summarization", model=SUMMARIZER_MODEL, truncation=True, device=-1)
        except Exception as e:
            raise RuntimeError(f"No se puede inicializar pipeline de transformers: {e}")

def summarize_with_transformers(text):
    _init_transformer_pipeline_if_needed()
    summarizer = _transformer_pipeline
    chunks = chunk_text_by_chars(text, max_chars=CHUNK_MAX_CHARS)
    summaries = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=250, min_length=60)[0]["summary_text"]
        summaries.append(out.strip())
    if len(summaries) == 1:
        return summaries[0]
    combined = " ".join(summaries)
    final = summarizer(combined, max_length=300, min_length=120)[0]["summary_text"]
    return final

# Hybrid: extractive -> abstractive
def summarize_hybrid(text):
    # extraer por chunks usando sumy para reducir volumen
    chunks = chunk_text_by_chars(text, max_chars=CHUNK_MAX_CHARS)
    reduced_parts = []
    per_chunk_sentences = max(1, EXTRACTIVE_SENTENCES // max(1, len(chunks)))
    for c in chunks:
        reduced_parts.append(summarize_with_sumy(c, sentences_count=per_chunk_sentences))
    reduced = " ".join(reduced_parts)
    # pulir con transformers (puede fallar si no instalado o falta memoria)
    return summarize_with_transformers(reduced)

# Wrapper público con fallback
def summarize_text(text, title):
    if not text or not text.strip():
        return "No hay transcripción para resumir."

    method = SUMMARIZER_METHOD
    safe_print(f"Generando resumen (método={method})...")
    resumen = None
    try:
        if method == "extractive":
            resumen = summarize_with_sumy(text)
        elif method == "abstractive":
            resumen = summarize_with_transformers(text)
        else:
            resumen = summarize_hybrid(text)
    except Exception as e:
        safe_print("⚠️ Error en summarizer avanzado:", str(e))
        safe_print("↪ Intentando fallback extractive (sumy)...")
        try:
            resumen = summarize_with_sumy(text)
        except Exception as e2:
            safe_print("❌ Fallback con sumy falló:", str(e2))
            safe_print("↪ Usando fallback sencillo (primeras frases).")
            resumen = simple_fallback_summary(text, max_sentences=6, max_chars=1200)


    header = f"Resumen del vídeo '{title}':\n\n"
    final = header + resumen.strip()
    # Limitar tamaño para WhatsApp (dividimos en mensajes si hace falta)
    if len(final) > MAX_WHATSAPP_CHARS:
        # no truncamos aquí; dejar que la función de envío divida en chunks
        return final
    return final

# ================= ENVIAR WHATSAPP (Twilio) =================

def send_whatsapp_message(text):
    if not TWILIO_SID or not TWILIO_TOKEN:
        raise RuntimeError("TWILIO_SID/TWILIO_TOKEN no configurados en env.")
    client = Client(TWILIO_SID, TWILIO_TOKEN)

    # dividir en partes si es necesario
    parts = []
    if len(text) <= MAX_WHATSAPP_CHARS:
        parts = [text]
    else:
        # dividir por saltos de línea preferiblemente
        remaining = text
        while remaining:
            if len(remaining) <= MAX_WHATSAPP_CHARS:
                parts.append(remaining)
                break
            # intentar cortar por doble salto de línea o punto
            cut_at = remaining.rfind("\n\n", 0, MAX_WHATSAPP_CHARS)
            if cut_at == -1:
                cut_at = remaining.rfind(".", 0, MAX_WHATSAPP_CHARS)
            if cut_at == -1 or cut_at < MAX_WHATSAPP_CHARS // 2:
                cut_at = MAX_WHATSAPP_CHARS
            parts.append(remaining[:cut_at].strip())
            remaining = remaining[cut_at:].lstrip()

    sids = []
    for p in parts:
        msg = client.messages.create(from_=TWILIO_WHATSAPP_FROM, to=TWILIO_WHATSAPP_TO, body=p)
        safe_print("Mensaje enviado:", msg.sid)
        sids.append(msg.sid)
    return sids

# ================= FLUJO PRINCIPAL =================

def main():
    safe_print("Inicio rutina: buscar último vídeo...")
    try:
        video_id, title = get_latest_video_id(YOUTUBER)
        safe_print("Último vídeo detectado:", title, "(", video_id, ")")

        if not is_new_video(video_id):
            safe_print("No hay vídeo nuevo. Fin.")
            return

        safe_print("Nuevo vídeo: obteniendo transcripción...")
        transcript = get_transcript(video_id)

        safe_print("Generando resumen...")
        resumen = summarize_text(transcript, title)

        safe_print("Enviando resumen por WhatsApp...")
        send_whatsapp_message(resumen)

        save_last_video(video_id)
        safe_print("✅ Proceso completado correctamente.")
    except Exception as e:
        safe_print("❌ Error en ejecución:", repr(e))

if __name__ == "__main__":
    main()
