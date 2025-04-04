from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse

app = FastAPI()

@app.get("/set-cookie")
def set_cookie():
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="mon_cookie", value="bonjour", max_age=3600, httponly=True)
    return response

@app.get("/")
def read_cookie(request: Request):
    valeur = request.cookies.get("mon_cookie")
    html_content = f"""
    <html>
        <body>
            <h1>Valeur du cookie : {valeur}</h1>
            <a href="/set-cookie">Définir le cookie</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
