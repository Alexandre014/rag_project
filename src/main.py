from datetime import datetime
from typing import List, Tuple
from uuid import uuid4
import requests
import asyncio

from nicegui import ui

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "deepseek-r1:32b" 
INDEX_PATH = "indexes/pdf_indexes/e5base"  # FAISS index path
OLLAMA_SERVER_URL = "https://tigre.loria.fr:11434/api/chat" 

messages: List[Tuple[str, str, str, str]] = []
first_message = """
Salut l'ami, je suis un **chatbot** pouvant répondre à tes questions en rapport avec le **BUT Informatique**.  
J'essaie de répondre en me basant sur les documents officiels que l'on m'a fourni.  
**Attention** : Il peut toute fois m'arriver de faire des erreurs.
"""
messages.append(("bot", "https://robohash.org/bot?bgset=bg2", first_message, datetime.now().strftime('%X')))


processing = False

def send_message(user_id: str, avatar: str, text: str) -> None:
    global processing
    if processing:
        return
    processing = True
    stamp = datetime.now().strftime('%X')
    messages.append((user_id, avatar, text, stamp))
    messages.append(("bot", "https://robohash.org/bot?bgset=bg2", "chargement...", datetime.now().strftime('%X')))
    chat_messages.refresh()
    
    text_input.disable
    send_button.disable
    
    async def fetch_reply():
        global processing
        try:
            response = await asyncio.to_thread(requests.post,
                API_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": text}],
                    "index_path": INDEX_PATH,
                    "ollama_server_url": OLLAMA_SERVER_URL,
                },
            )
            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]
            else:
                reply = "Erreur : Impossible d'obtenir une réponse."
        except Exception as e:
            reply = f"Erreur : {str(e)}"

        messages.pop()
        messages.append(("bot", "https://robohash.org/bot?bgset=bg2", reply, datetime.now().strftime('%X')))
        chat_messages.refresh()
        
        text_input.enable
        send_button.enable
        processing = False

    asyncio.create_task(fetch_reply())

@ui.refreshable
def chat_messages() -> None:
    if messages:
        for user_id, avatar, text, stamp in messages:
            with ui.chat_message(avatar=avatar, stamp=stamp, sent=user_id != "bot"):
                ui.markdown(text)
    else:
        ui.label("Aucun message pour le moment").classes("mx-auto my-36")
    ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")



@ui.page("/")
async def main():
    global text_input, send_button
    user_id = str(uuid4())
    avatar = f"https://robohash.org/{user_id}?bgset=bg2"

    def send():
        if text_input.value.strip():
            send_message(user_id, avatar, text_input.value)
            text_input.value = ""

    ui.add_css("a:link, a:visited {color: inherit !important; text-decoration: underline; font-weight: 500}")
    with ui.footer().classes("bg-white"), ui.column().classes("w-full max-w-3xl mx-auto my-6"):
        with ui.row().classes("w-full no-wrap items-center"):
            ui.image(avatar).classes("w-10 h-10 rounded-full")
            text_input = ui.input(placeholder="Posez votre question...").on("keydown.enter", send)\
                .props("rounded outlined input-class=mx-3").classes("flex-grow")
            send_button = ui.button("Envoyer", on_click=send).classes("ml-2")
    
    await ui.context.client.connected()
    with ui.column().classes("w-full max-w-2xl mx-auto items-stretch"):
        chat_messages()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8080, reconnect_timeout=2000)
