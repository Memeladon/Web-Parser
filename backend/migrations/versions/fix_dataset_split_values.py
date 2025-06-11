"""fix dataset split values

Revision ID: fix_dataset_split_values
Revises: 0e5d3ad97d74
Create Date: 2024-03-19

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'fix_dataset_split_values'
down_revision = '0e5d3ad97d74'
branch_labels = None
depends_on = None

def upgrade():
    # Update 'unassigned' to 'UNASSIGNED'
    op.execute("UPDATE article SET dataset_split = 'UNASSIGNED' WHERE dataset_split = 'unassigned'")
    
    # Update 'train' to 'TRAIN'
    op.execute("UPDATE article SET dataset_split = 'TRAIN' WHERE dataset_split = 'train'")
    
    # Update 'validation' to 'VALIDATION'
    op.execute("UPDATE article SET dataset_split = 'VALIDATION' WHERE dataset_split = 'validation'")
    
    # Update 'test' to 'TEST'
    op.execute("UPDATE article SET dataset_split = 'TEST' WHERE dataset_split = 'test'")

def downgrade():
    # Update 'UNASSIGNED' to 'unassigned'
    op.execute("UPDATE article SET dataset_split = 'unassigned' WHERE dataset_split = 'UNASSIGNED'")
    
    # Update 'TRAIN' to 'train'
    op.execute("UPDATE article SET dataset_split = 'train' WHERE dataset_split = 'TRAIN'")
    
    # Update 'VALIDATION' to 'validation'
    op.execute("UPDATE article SET dataset_split = 'validation' WHERE dataset_split = 'VALIDATION'")
    
    # Update 'TEST' to 'test'
    op.execute("UPDATE article SET dataset_split = 'test' WHERE dataset_split = 'TEST'") 