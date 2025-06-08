from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column
from ..dependencies import Base

class Article(Base):
    __tablename__ = "article"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(nullable=False)
    author: Mapped[str] = mapped_column(nullable=False)
    abstract: Mapped[str] = mapped_column()
    content: Mapped[str] = mapped_column(nullable=False)
    source_url: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
