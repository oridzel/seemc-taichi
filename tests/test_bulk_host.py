import pytest
from seemc_taichi.bulk import BulkPhysicsConfig


def test_bulk_config_defaults_validate():
    BulkPhysicsConfig().validate()


def test_bulk_config_rejects_too_small_q_grid():
    with pytest.raises(ValueError, match="n_q_sample"):
        BulkPhysicsConfig(n_q_sample=4).validate()


def test_browning_requires_z():
    with pytest.raises(ValueError, match="atomic_number"):
        BulkPhysicsConfig(elastic_low_energy_model="browning", atomic_number=0.0).validate()
