import sqlite3
import bcrypt
from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from nicegui import app, ui
from uuid import uuid4
from datetime import datetime
from typing import List, Tuple
import requests
import asyncio
from fastapi.middleware.cors import CORSMiddleware

conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    username TEXT NOT NULL
)
""")
conn.commit()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def get_user(username: str):
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()

def create_user(username: str, password: str):
    if not get_user(username):
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()

# authentication middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        excluded_paths = [
            "/login",
            "/register",
            "/_nicegui",
            "/_nicegui_ws",
            "/favicon.ico",  # Add favicon to excluded paths
        ]
        path = request.url.path
        if any(path.startswith(excluded) for excluded in excluded_paths):
            print("exempted")
            return await call_next(request)
        print("middleware")

        # Check if user is stored in app.storage.user
        if 'username' in app.storage.user:
            print("user found in storage")
            return await call_next(request)
        
        print("bad")
        return RedirectResponse("/login")

app.add_middleware(AuthMiddleware)

@ui.page("/register")
def register_page():
    def register():
        if username.value and password.value:
            create_user(username.value, password.value)
            ui.notify("Inscription réussie !", color="positive")
            print("inscription ok")
            ui.navigate.to("/login")
        else:
            ui.notify("Veuillez remplir tous les champs", color="negative")
    
    with ui.card().classes("absolute-center"):
        username = ui.input("Nom d'utilisateur")
        password = ui.input("Mot de passe", password=True)
        ui.button("S'inscrire", on_click=register)

@ui.page("/login")
def login_page():
    def login():
        user = get_user(username.value)
        if user and verify_password(password.value, user[2]):
            # Store user info in app.storage instead of cookies
            app.storage.user['username'] = user[1]
            app.storage.user['session_id'] = str(uuid4())  # Optional: Store a session ID in the storage
            ui.navigate.to("/")
            #response = RedirectResponse("/", status_code=303)
            print("redirect response passed")
        else:
            ui.notify("Identifiants incorrects", color="negative")
    
    with ui.card().classes("absolute-center"):
        username = ui.input("Nom d'utilisateur")
        password = ui.input("Mot de passe", password=True)
        ui.button("Se connecter", on_click=login)

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "deepseek-r1:32b" 
INDEX_PATH = "indexes/pdf_indexes/e5base"  # FAISS index path
OLLAMA_SERVER_URL = "https://tigre.loria.fr:11434/api/chat" 

messages: List[Tuple[str, str, str, str]] = []
first_message = """
Salut l'ami, je suis un **chatbot** pouvant répondre à tes questions en rapport avec le **BUT Informatique**.  
J'essaie de répondre en me basant sur les documents officiels que l'on m'a fourni.  
**Attention** : Il peut toutefois m'arriver de faire des erreurs.
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
async def main_page():
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
    ui.run(storage_secret="secret_secure_storage", port=8080, reconnect_timeout=2000)
