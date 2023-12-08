"""make country non-nullable

Revision ID: e302ed8f61c1
Revises: 
Create Date: 2023-11-08 19:58:28.215366

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e302ed8f61c1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('country',
               existing_type=sa.VARCHAR(length=64),
               nullable=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('country',
               existing_type=sa.VARCHAR(length=64),
               nullable=True)

    # ### end Alembic commands ###