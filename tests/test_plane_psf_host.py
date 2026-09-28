import numpy as np

from seemc_taichi.plane_psf import (
    birth_radius_depth_nm,
    classify_emissions,
    emission_xy_nm,
    population_summary,
    radial_profiles,
    seed_radius_depth_summary,
)


def synthetic_events():
    # 0 direct reflection
    # 1 ballistic single-origin SE
    # 2 single-origin SE that elastically scatters after birth
    # 3 single-origin SE with a post-birth inelastic collision
    # 4 generation-1 SE born after parent elastic scattering
    # 5 generation-1 SE born after a prior primary inelastic collision
    # 6 generation-2 SE
    # 7 low-energy primary emission
    # 8 one-elastic BSE
    # 9 plural-elastic BSE
    # 10 elastic+inelastic BSE
    # 11 high-energy cascade BSE
    # 12 primary BSE with inelastic-only transport
    # 13 primary BSE with no bulk collision
    n = 14
    return {
        "emission_energy_ev": np.array(
            [10000, 5, 8, 7, 4, 6, 3, 20, 9000, 8000, 7000, 100, 5000, 10000],
            float,
        ),
        "emission_x_angstrom": 10.0 * np.arange(n, dtype=float),
        "emission_y_angstrom": np.zeros(n),
        "emission_generation": np.array([0, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0]),
        "emission_mechanism": np.array([2] + [1] * (n - 1)),
        "emission_own_elastic_count": np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 3, 2, 0, 0, 0]),
        "emission_own_inelastic_count": np.array([0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 2, 0]),
        "emission_path_elastic_count": np.array([0, 0, 1, 0, 1, 0, 0, 1, 1, 3, 2, 2, 0, 0]),
        "emission_path_inelastic_count": np.array([0, 1, 1, 2, 1, 2, 2, 3, 0, 0, 1, 1, 2, 0]),
        "emission_birth_x_angstrom": np.array(
            [0, 0, 0, 0, 2000, 0, 100, 0, 0, 0, 0, 500, 0, 0], float
        ),
        "emission_birth_y_angstrom": np.zeros(n),
        "emission_birth_z_angstrom": np.array(
            [0, 1, 2, 3, 10, 20, 5, 0, 0, 0, 0, 400, 0, 0], float
        ),
    }


def test_population_partitions_and_definitions():
    data = synthetic_events()
    masks = classify_emissions(data, cutoff_ev=50.0)

    assert np.flatnonzero(masks["direct_barrier_reflection"]).tolist() == [0]
    assert np.flatnonzero(masks["se_total"]).tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert np.flatnonzero(masks["se_single_inelastic_origin"]).tolist() == [1, 2, 3]
    assert np.flatnonzero(masks["se_strict_single_inelastic"]).tolist() == [1]
    assert np.flatnonzero(masks["se_multiple_origin"]).tolist() == [4, 5, 6, 7]
    assert np.flatnonzero(masks["se_mult_gen1_after_elastic"]).tolist() == [4]
    assert np.flatnonzero(masks["se_mult_gen1_after_prior_inelastic"]).tolist() == [5]
    assert np.flatnonzero(masks["se_mult_generation_2plus"]).tolist() == [6]
    assert np.flatnonzero(masks["se_mult_primary"]).tolist() == [7]
    assert np.flatnonzero(masks["se1_current_rule"]).tolist() == [1, 2, 4, 5]
    assert np.flatnonzero(masks["se2_current_rule"]).tolist() == [3, 6, 7]
    assert np.flatnonzero(masks["bse_total"]).tolist() == [8, 9, 10, 11, 12, 13]
    assert np.flatnonzero(masks["bse_single_elastic"]).tolist() == [8]
    assert np.flatnonzero(masks["bse_plural_elastic_only"]).tolist() == [9]
    assert np.flatnonzero(masks["bse_elastic_plus_inelastic"]).tolist() == [10]
    assert np.flatnonzero(masks["bse_other"]).tolist() == [11, 12, 13]
    assert np.flatnonzero(masks["bse_other_high_energy_cascade"]).tolist() == [11]
    assert np.flatnonzero(masks["bse_other_primary_inelastic_only"]).tolist() == [12]
    assert np.flatnonzero(masks["bse_other_primary_no_collision"]).tolist() == [13]
    assert np.array_equal(
        masks["transport_total"],
        masks["combined_single"] | masks["combined_multiple"],
    )


def test_annular_psf_integrates_to_yield():
    data = synthetic_events()
    masks = classify_emissions(data, cutoff_ev=50.0)
    x_nm, y_nm = emission_xy_nm(data)
    radius = np.hypot(x_nm, y_nm)
    edges = np.array([0.0, 1.5, 5.5, 20.0])
    profiles = radial_profiles(radius, masks, edges, n_primaries=20)

    area = profiles["annulus_area_nm2"]
    for name, mask in masks.items():
        integrated = float(np.sum(profiles[f"psf__{name}"] * area))
        assert np.isclose(integrated, np.count_nonzero(mask) / 20.0)


def test_gaussian_convolution_is_deterministic_and_summary_is_normalized():
    data = synthetic_events()
    x1, y1 = emission_xy_nm(data, beam_sigma_nm=1.0, seed=17)
    x2, y2 = emission_xy_nm(data, beam_sigma_nm=1.0, seed=17)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)

    masks = classify_emissions(data, cutoff_ev=50.0)
    rows = population_summary(data, np.hypot(x1, y1), masks, n_primaries=20)
    lookup = {row["population"]: row for row in rows}
    assert np.isclose(
        lookup["se_single_inelastic_origin"]["fraction_of_family"], 3.0 / 7.0
    )
    assert np.isclose(lookup["bse_single_elastic"]["fraction_of_family"], 1.0 / 6.0)
    assert np.isclose(
        lookup["se_mult_gen1_after_elastic"]["fraction_of_family"], 1.0 / 4.0
    )


def test_seed_radius_depth_summary_and_relative_poisson_sem():
    data = synthetic_events()
    masks = classify_emissions(data, cutoff_ev=50.0)
    birth_radius, birth_depth = birth_radius_depth_nm(data)
    assert np.isclose(birth_radius[4], 200.0)
    assert np.isclose(birth_depth[4], 1.0)

    rows = seed_radius_depth_summary(data, masks)
    lookup = {row["population"]: row for row in rows}
    assert lookup["se_single_inelastic_origin"]["count"] == 3
    assert lookup["se_multiple_origin"]["count"] == 4
    assert np.isclose(
        lookup["se_multiple_origin"]["fraction_shallow_le_2nm_and_far_ge_100nm"],
        0.25,
    )

    radius = np.hypot(*emission_xy_nm(data))
    profiles = radial_profiles(
        radius, masks, np.array([0.0, 1.5, 5.5, 20.0]), n_primaries=20
    )
    count = profiles["count__se_total"]
    relative = profiles["relative_poisson_sem__se_total"]
    assert np.allclose(relative[count > 0], 1.0 / np.sqrt(count[count > 0]))
    assert np.all(np.isnan(relative[count == 0]))
