"""initial migration

Revision ID: 000000000001
Revises:
Create Date: 2026-06-21 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = '000000000001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'patient_alias',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('anonymous_id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_patient_alias_anonymous_id'), 'patient_alias', ['anonymous_id'], unique=True)
    op.create_index(op.f('ix_patient_alias_id'), 'patient_alias', ['id'], unique=False)

    op.create_table(
        'plan',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('plan_uid', sa.String(length=128), nullable=False),
        sa.Column('plan_name', sa.String(length=256), nullable=True),
        sa.Column('modality', sa.String(length=32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patient_alias.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_plan_id'), 'plan', ['id'], unique=False)
    op.create_index(op.f('ix_plan_plan_uid'), 'plan', ['plan_uid'], unique=True)

    op.create_table(
        'beam',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('beam_name', sa.String(length=128), nullable=False),
        sa.Column('beam_number', sa.Integer(), nullable=False),
        sa.Column('beam_type', sa.String(length=64), nullable=True),
        sa.Column('energy', sa.String(length=32), nullable=True),
        sa.Column('control_points_data', sa.JSON(), nullable=False),
        sa.Column('leaf_positions', sa.JSON(), nullable=False),
        sa.Column('dose_rates', sa.JSON(), nullable=True),
        sa.Column('gantry_angles', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['plan.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_beam_id'), 'beam', ['id'], unique=False)

    op.create_table(
        'qa_result',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('beam_id', sa.Integer(), nullable=False),
        sa.Column('log_filename', sa.String(length=256), nullable=True),
        sa.Column('max_leaf_deviation_mm', sa.Float(), nullable=True),
        sa.Column('mean_leaf_deviation_mm', sa.Float(), nullable=True),
        sa.Column('rmse_mm', sa.Float(), nullable=True),
        sa.Column('dose_rate_deviation_pct', sa.Float(), nullable=True),
        sa.Column('control_point_pass_rate_pct', sa.Float(), nullable=True),
        sa.Column('num_control_points', sa.Integer(), nullable=True),
        sa.Column('num_failed_control_points', sa.Integer(), nullable=True),
        sa.Column('num_leaves', sa.Integer(), nullable=True),
        sa.Column('gantry_angle_range', sa.String(length=64), nullable=True),
        sa.Column('qa_date', sa.DateTime(), default=sa.func.now()),
        sa.Column('overall_pass', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['beam_id'], ['beam.id'], ),
        sa.ForeignKeyConstraint(['plan_id'], ['plan.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_qa_result_id'), 'qa_result', ['id'], unique=False)

    op.create_table(
        'leaf_error_sample',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('qa_result_id', sa.Integer(), nullable=False),
        sa.Column('control_point_index', sa.Integer(), nullable=True),
        sa.Column('leaf_index', sa.Integer(), nullable=True),
        sa.Column('bank', sa.String(length=16), nullable=True),
        sa.Column('planned_position_mm', sa.Float(), nullable=True),
        sa.Column('actual_position_mm', sa.Float(), nullable=True),
        sa.Column('deviation_mm', sa.Float(), nullable=True),
        sa.Column('timestamp_sec', sa.Float(), nullable=True),
        sa.Column('log_time', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['qa_result_id'], ['qa_result.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_leaf_error_sample_id'), 'leaf_error_sample', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_leaf_error_sample_id'), table_name='leaf_error_sample')
    op.drop_table('leaf_error_sample')
    op.drop_index(op.f('ix_qa_result_id'), table_name='qa_result')
    op.drop_table('qa_result')
    op.drop_index(op.f('ix_beam_id'), table_name='beam')
    op.drop_table('beam')
    op.drop_index(op.f('ix_plan_plan_uid'), table_name='plan')
    op.drop_index(op.f('ix_plan_id'), table_name='plan')
    op.drop_table('plan')
    op.drop_index(op.f('ix_patient_alias_id'), table_name='patient_alias')
    op.drop_index(op.f('ix_patient_alias_anonymous_id'), table_name='patient_alias')
    op.drop_table('patient_alias')
