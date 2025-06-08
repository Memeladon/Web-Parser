# schemas/article.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional

# Базовые схемы
class ArticleBase(BaseModel):
    title: str
    author: str
    abstract: Optional[str] = ""
    content: str
    source_url: str

# Схемы для ответов
class ArticleResponse(ArticleBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Схемы для запросов
class ArticleCreate(ArticleBase):
    pass

class ArticleUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[str] = None
    content: Optional[str] = None
    source_url: Optional[str] = None