"""Add dataset_split column

Revision ID: add_dataset_split
Revises: fa287010cabc
Create Date: 2024-04-22 20:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_dataset_split'
down_revision: Union[str, None] = 'fa287010cabc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Add dataset_split column with default value 'unassigned'
    op.add_column('article', sa.Column('dataset_split', sa.String(), nullable=False, server_default='unassigned'))

def downgrade() -> None:
    op.drop_column('article', 'dataset_split') 