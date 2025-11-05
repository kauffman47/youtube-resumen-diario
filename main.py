import os
import json
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from twilio.rest import Client

# ==== CONFIGURACIÓN ====

# Nombre de usuario o ID del canal de YouTube que quieres seguir
YOUTUBER = "@JoseLuisCavatv"  # <- cámbialo por el canal que quieras (ej: "VisualPolitik")

# Ruta del archivo donde guardaremos el último video procesado
LAST_VIDEO_FILE = "last_video.json"

# Inicializar APIs
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# ==== 1. OBTENER ÚLTIMO VIDEO DEL CANAL ====

def get_latest_video_id(channel_identifier):
    """
    Devuelve el ID y título del último video del canal.
    Acepta tanto username (antiguo) como @handle o Channel ID.
    """
    # Si el canal es un handle (empieza con @)
    if channel_identifier.startswith("@"):
        search_response = youtube.search().list(
            part="snippet",
            q=channel_identifier,
            type="channel",
            maxResults=1
        ).execute()
        if not search_response.get("items"):
            raise ValueError("No se encontró canal con ese handle.")
        channel_id = search_response["items"][0]["snippet"]["channelId"]
    # Si es un Channel ID (empieza por UC)
    elif channel_identifier.startswith("UC"):
        channel_id = channel_identifier
    # Si es un username antiguo
    else:
        channel_response = youtube.channels().list(
            part="id",
            forUsername=channel_identifier
        ).execute()
        if not channel_response.get("items"):
            raise ValueError("No se encontró canal con ese nombre de usuario.")
        channel_id = channel_response["items"][0]["id"]

    # Obtener la playlist de subidas del canal
    channel_details = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    uploads_playlist_id = channel_details["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    playlist_items = youtube.playlistItems().list(
        playlistId=uploads_playlist_id,
        part="snippet",
        maxResults=1
    ).execute()

    video_id = playlist_items["items"][0]["snippet"]["resourceId"]["videoId"]
    title = playlist_items["items"][0]["snippet"]["title"]

    return video_id, title

# ==== 2. COMPROBAR SI ES NUEVO ====

def is_new_video(video_id):
    if not os.path.exists(LAST_VIDEO_FILE):
        return True

    with open(LAST_VIDEO_FILE, "r") as f:
        data = json.load(f)

    return data.get("last_video_id") != video_id

def save_last_video(video_id):
    with open(LAST_VIDEO_FILE, "w") as f:
        json.dump({"last_video_id": video_id}, f)

# ==== 3. DESCARGAR TRANSCRIPCIÓN ====

from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
        text = " ".join([t["text"] for t in transcript])
        return text
    except Exception as e:
        raise ValueError(f"No se pudo obtener la transcripción: {e}")



# ==== 4. RESUMIR CON OPENAI ====

def summarize_text(text, title):
    prompt = f"""
Haz un resumen estructurado y claro del siguiente texto, similar al ejemplo que te mostré antes.
Texto del video "{title}":
{text}
"""
    completion = client_openai.chat.completions.create(
        model="gpt-5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return completion.choices[0].message.content.strip()

# ==== 5. ENVIAR POR WHATSAPP ====

def send_whatsapp_message(text):
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_TOKEN")
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_="whatsapp:+14155238886",  # número del sandbox de Twilio
        to="whatsapp:+34665447902",     # <-- cambia por tu número (con prefijo internacional)
        body=text
    )
    print("Mensaje enviado:", message.sid)

# ==== 6. FLUJO PRINCIPAL ====

def main():
    print("Buscando nuevo video...")
    try:
        video_id, title = get_latest_video_id(YOUTUBER)
        print("Último video:", title)

        if not is_new_video(video_id):
            print("No hay video nuevo. Fin.")
            return

        print("Nuevo video detectado. Descargando transcripción...")
        transcript = get_transcript(video_id)

        print("Generando resumen con OpenAI...")
        resumen = summarize_text(transcript, title)

        print("Enviando resumen por WhatsApp...")
        send_whatsapp_message(f"🧠 Resumen del nuevo video '{title}':\n\n{resumen}")

        save_last_video(video_id)
        print("✅ Proceso completado.")

    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == "__main__":
    main()

