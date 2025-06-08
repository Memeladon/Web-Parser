import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase, declared_attr

from src.config import settings

DATABASE_URL = settings.get_db_url()
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not defined!")

engine = create_engine(
    url=DATABASE_URL
)
session = sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False)

def get_db() -> Session:
    db = session()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    """Базовый класс для всех моделей"""
    __abstract__ = True  # Класс абстрактный, чтобы не создавать отдельную таблицу для него

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower() + "s"

    def to_dict(self) -> dict:
        """Универсальный метод для конвертации объекта SQLAlchemy в словарь"""
        # Получаем маппер для текущей модели
        columns = class_mapper(self.__class__).columns
        # Возвращаем словарь всех колонок и их значений
        return {column.key: getattr(self, column.key) for column in columns}