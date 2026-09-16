import numpy as np

from seemc_taichi.trajectory_anim_cli import classify_electron_lineage, trapezoid_outline


def test_lineage_classification_current_se1_rule():
    electron = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
    generation = np.array([0, 0, 1, 1, 1, 1, 2, 2], np.int32)
    inelastic = np.array([0, 2, 0, 0, 0, 1, 0, 0], np.int32)
    classes = classify_electron_lineage(electron, generation, inelastic)
    assert classes[0] == 0  # primary
    assert classes[1] == 1  # generation 1, no own inelastic collision
    assert classes[2] == 2  # generation 1 but own inelastic collision
    assert classes[3] == 2  # generation >= 2


def test_trapezoid_outline_geometry():
    x, z = trapezoid_outline(50.0, 70.0, 50.0)
    assert np.allclose(x, [-35.0, -25.0, 25.0, 35.0, -35.0])
    assert np.allclose(z, [0.0, -50.0, -50.0, 0.0, 0.0])
