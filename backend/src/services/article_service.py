from typing import Optional, List
from sqlalchemy.orm import Session
from cachetools import TTLCache

from src.database.repositories import ArticleRepository
from src.database.models import Article
from src.schemas.article import ArticleCreate, ArticleResponse, ArticleUpdate


class ArticleService:
    def __init__(self, session: Session):
        self.session = session
        self.cache = TTLCache(maxsize=100, ttl=300)  # Кэш на 5 минут

    def create_article(self, article_data: ArticleCreate) -> ArticleResponse:

        """Создание статьи с валидацией и преобразованием"""
        try:
            db_article = ArticleRepository.create(
                self.session,
                article_data.model_dump()  # Используем model_dump() вместо dict()
            )
            return ArticleResponse.model_validate(db_article)  # Замена from_orm()
        except Exception as e:
            raise RuntimeError(f"|SERVICE| Ошибка создания: {str(e)}") from e

    def get_article(self, article_id: int) -> ArticleResponse:
        """Получение статьи с преобразованием"""
        try:
            db_article = ArticleRepository.get_by_id(self.session, article_id)
            return ArticleResponse.model_validate(db_article)
        except LookupError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise RuntimeError(f"Ошибка получения: {str(e)}") from e

    def get_all_articles(self, skip: int = 0, limit: int = 100) -> List[ArticleResponse]:
        """Получение статей с кэшированием"""
        try:
            if "all_articles" in self.cache:
                return self.cache["all_articles"]

            articles = ArticleRepository.get_all(self.session, skip, limit)
            result = [ArticleResponse.model_validate(a) for a in articles]
            self.cache["all_articles"] = result
            return result

        except Exception as e:
            raise RuntimeError(f"Ошибка получения списка: {str(e)}") from e

    def update_article(self, article_id: int, update_data: ArticleUpdate) -> ArticleResponse:
        """Обновление статьи"""
        try:
            # Фильтрация None значений для частичного обновления
            filtered_data = update_data.dict(exclude_unset=True)

            db_article = ArticleRepository.update(
                self.session,
                article_id,
                filtered_data
            )
            return ArticleResponse.model_validate(db_article)
        except LookupError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise RuntimeError(f"Ошибка обновления: {str(e)}") from e

    def delete_article(self, article_id: int) -> None:
        """Удаление статьи"""
        try:
            ArticleRepository.delete(self.session, article_id)
        except LookupError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise RuntimeError(f"Ошибка удаления: {str(e)}") from e