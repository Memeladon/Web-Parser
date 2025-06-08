from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List
import asyncio

from src.database.dependencies import get_db
from src.services.article_service import ArticleService
from src.schemas.article import ArticleResponse, ArticleCreate, ArticleUpdate
from src.scrapers.scraper import run_scraper

router = APIRouter(tags=['articles'], prefix='/articles')
templates = Jinja2Templates(directory="public/templates/")

@router.post(
    "/",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать новую статью"
)
async def create_article(
    article_data: ArticleCreate,
    db: Session = Depends(get_db)
):
    """Создание новой статьи с валидацией данных"""
    try:
        return ArticleService(db).create_article(article_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"|HANDLERS| Ошибка при создании статьи: {str(e)}"
        )

@router.get("/view")
async def view_articles(request: Request, db=Depends(get_db)):
    articles = ArticleService(db).get_all_articles(skip=0, limit=100)
    return templates.TemplateResponse("articles.html", {
        "request": request,
        "articles": articles
    })

@router.get(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Получить статью по ID"
)
async def get_article(
    article_id: int,
    db: Session = Depends(get_db)
):
    """Получение конкретной статьи по её идентификатору"""
    try:
        return ArticleService(db).get_article(article_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статьи: {str(e)}"
        )

@router.get(
    "/",
    response_model=List[ArticleResponse],
    summary="Получить список статей"
)
async def get_articles(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Получение списка статей с пагинацией"""
    try:
        return ArticleService(db).get_all_articles(skip=skip, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении списка статей: {str(e)}"
        )

@router.put(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Обновить статью"
)
async def update_article(
    article_id: int,
    update_data: ArticleUpdate,
    db: Session = Depends(get_db)
):
    """Обновление существующей статьи"""
    try:
        return ArticleService(db).update_article(article_id, update_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении статьи: {str(e)}"
        )

@router.delete(
    "/{article_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить статью"
)
async def delete_article(
    article_id: int,
    db: Session = Depends(get_db)
):
    """Удаление статьи по идентификатору"""
    try:
        ArticleService(db).delete_article(article_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении статьи: {str(e)}"
        )
    return {"message": "Статья успешно удалена"}

@router.post("/scrape/stackexchange", summary="Trigger StackExchange scraping in background")
async def trigger_stackexchange_scrape(background_tasks: BackgroundTasks):
    background_tasks.add_task(asyncio.run, run_scraper())
    return {"message": "StackExchange scraping started in background"}

