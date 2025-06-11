"""merge heads

Revision ID: 0e5d3ad97d74
Revises: 07118f1eec08, add_dataset_split
Create Date: 2025-06-10 02:40:09.628865

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0e5d3ad97d74'
down_revision: Union[str, None] = ('07118f1eec08', 'add_dataset_split')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
