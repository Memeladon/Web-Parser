from datetime import datetime
from enum import Enum
from sqlalchemy.orm import Mapped, mapped_column
from ..dependencies import Base

class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNASSIGNED = "unassigned"

class Article(Base):
    __tablename__ = "article"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(nullable=False)
    author: Mapped[str] = mapped_column(nullable=False)
    abstract: Mapped[str] = mapped_column()
    content: Mapped[str] = mapped_column(nullable=False)
    source_url: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    cleaned: Mapped[bool] = mapped_column(nullable=False, default=False, server_default='0')
    dataset_split: Mapped[DatasetSplit] = mapped_column(
        nullable=False,
        default=DatasetSplit.UNASSIGNED,
        server_default=DatasetSplit.UNASSIGNED
    )
