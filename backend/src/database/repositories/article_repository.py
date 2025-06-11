from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, NoResultFound, IntegrityError
from typing import Optional, List, Type
from datetime import datetime

from src.database.models import Article

class ArticleRepository:
    @staticmethod
    def create(session: Session, article_data: dict) -> Article:
        """Создание статьи"""
        try:
            article = Article(
                title=article_data["title"],
                author=article_data["author"],
                abstract=article_data.get("abstract", ""),
                content=article_data["content"],
                source_url=article_data["source_url"],
                created_at=article_data.get("created_at", datetime.now()),
            )
            session.add(article)
            session.commit()
            session.refresh(article)
            return article
        except IntegrityError as e:
            session.rollback()
            raise RuntimeError(f"|REPOSITORY| Ошибка целостности данных: {str(e)}") from e
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"|REPOSITORY| Ошибка базы данных: {str(e)}") from e

    @staticmethod
    def get_by_id(session: Session, article_id: int) -> Article:
        """Получение статьи по ID"""
        try:
            return session.query(Article).filter(Article.id == article_id).one()
        except NoResultFound as e:
            raise LookupError(f"|REPOSITORY| Статья с ID {article_id} не найдена") from e
        except SQLAlchemyError as e:
            raise RuntimeError(f"|REPOSITORY| Ошибка базы данных: {str(e)}") from e

    @staticmethod
    def get_all(session: Session, skip: int = 0, limit: int = 100) -> list[Type[Article]]:
        """Получение списка статей"""
        try:
            return session.query(Article).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise RuntimeError(f"|REPOSITORY| Ошибка базы данных: {str(e)}") from e

    @staticmethod
    def update(session: Session, article_id: int, update_data: dict) -> Article:
        """Обновление статьи"""
        try:
            article = session.query(Article).filter(Article.id == article_id).one()
            for key, value in update_data.items():
                if hasattr(article, key):
                    setattr(article, key, value)
            session.commit()
            session.refresh(article)
            return article
        except NoResultFound as e:
            raise LookupError(f"Статья с ID {article_id} не найдена") from e
        except IntegrityError as e:
            session.rollback()
            raise RuntimeError(f"|REPOSITORY| Ошибка целостности данных: {str(e)}") from e
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"|REPOSITORY| Ошибка базы данных: {str(e)}") from e

    @staticmethod
    def delete(session: Session, article_id: int) -> None:
        """Удаление статьи"""
        try:
            article = session.query(Article).filter(Article.id == article_id).one()
            session.delete(article)
            session.commit()
        except NoResultFound as e:
            raise LookupError(f"|REPOSITORY| Статья с ID {article_id} не найдена") from e
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"|REPOSITORY| Ошибка базы данных: {str(e)}") from e

    @staticmethod
    def get_by_source_url(session: Session, source_url: str) -> Optional[Article]:
        """Получение статьи по source_url"""
        try:
            return session.query(Article).filter(Article.source_url == source_url).one_or_none()
        except SQLAlchemyError as e:
            raise RuntimeError(f"|REPOSITORY| Ошибка базы данных: {str(e)}") from e