import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from .config import settings
from .handlers import article_router

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION, root_path="/api")


# ---------- Промежуточное ПО ---------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Маршрутизация ------------ #
app.include_router(article_router)


@app.get("/")
def home():
    # проверка работоспособности
    return {"msg": "ok"}
