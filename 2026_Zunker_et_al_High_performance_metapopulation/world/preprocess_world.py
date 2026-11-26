#!/usr/bin/env python3
"""Build the inputs of the worldwide application.

Patches are the cells of the 1-degree population grid GPW v4.11 (2020) with at least
1,000 inhabitants. The static mobility matrix follows a gravity model, where each
patch sends a proportion mu = 0.1 of its inhabitants to the other patches with weights
N_q / d_pq^2, where d_pq is the great circle distance between the cell centres.
Each area is divided into the six age groups of the simulation scenario (Table C.2),
and the epidemic begins with infectious individuals in the Frankfurt cell.

"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = HERE / "gpw_v4_population_count_rev11_2020_1_deg.asc"
MIN_POPULATION = 1000.0
MOBILE_FRACTION = 0.1
GROUPS = 6
SEED_INFECTED = 100.0
FRANKFURT = (50.11, 8.68)  # latitude, longitude
EARTH_RADIUS_KM = 6371.0


def read_grid(path):
    header = {}
    with open(path) as stream:
        for _ in range(6):
            key, value = stream.readline().split()
            header[key.lower()] = float(value)
        values = np.loadtxt(stream)
    values[values == header["nodata_value"]] = 0.0
    rows, cols = values.shape
    size = header["cellsize"]
    lat = header["yllcorner"] + (rows - 0.5 - np.arange(rows)) * size
    lon = header["xllcorner"] + (np.arange(cols) + 0.5) * size
    return values, lat, lon


def great_circle(lat_a, lon_a, lat_b, lon_b):
    lat_a, lon_a, lat_b, lon_b = map(np.radians, (lat_a, lon_a, lat_b, lon_b))
    a = (np.sin((lat_b - lat_a) / 2) ** 2
         + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def gravity_matrix(population, lat, lon):
    p = population.size
    h = np.empty((p, p))
    for origin in range(p):
        d = great_circle(lat[origin], lon[origin], lat, lon)
        weight = population / np.maximum(d, 1e-9) ** 2
        weight[origin] = 0.0
        h[origin] = MOBILE_FRACTION * weight / weight.sum()
        h[origin, origin] = 1.0 - MOBILE_FRACTION
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=GRID, help="GPW v4.11 1-degree population count grid (ASCII)")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()

    grid, lat_axis, lon_axis = read_grid(args.grid)
    rows, cols = np.nonzero(grid >= MIN_POPULATION)
    population = grid[rows, cols]
    lat, lon = lat_axis[rows], lon_axis[cols]
    p = population.size

    seed_row = int((90.0 - FRANKFURT[0]) // 1)
    seed_col = int((FRANKFURT[1] + 180.0) // 1)
    seed = np.flatnonzero((rows == seed_row) & (cols == seed_col))
    if seed.size != 1:
        raise RuntimeError("The Frankfurt cell is not among the selected patches.")
    seed = int(seed[0])

    # Age shares of Table C.2: (0.9 + 0.2 i / (N_G + 1)) / N_G, which sum to one.
    shares = (0.9 + 0.2 * np.arange(1, GROUPS + 1) / (GROUPS + 1)) / GROUPS
    group_population = shares[:, None] * population[None, :]
    initial = np.zeros((GROUPS, 4, p))
    initial[:, 0, :] = group_population
    initial[:, 2, seed] = SEED_INFECTED * shares
    initial[:, 0, seed] -= SEED_INFECTED * shares

    h = gravity_matrix(population, lat, lon)
    assert np.allclose(h.sum(axis=1), 1.0)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "world_inputs.bin", "wb") as stream:
        np.array([p, GROUPS], dtype="<i4").tofile(stream)
        group_population.astype("<f8").tofile(stream)
        initial.reshape(4 * GROUPS, p).astype("<f8").tofile(stream)
        h.astype("<f8").tofile(stream)
    np.savez(args.output / "world_cells.npz", row=rows, col=cols, lat=lat, lon=lon, population=population,
             seed=seed)
    print(f"{p} patches, {population.sum():.4e} inhabitants ({population.sum() / grid.sum():.1%} of the grid), "
          f"seed patch {seed} at ({lat[seed]}, {lon[seed]}) with {population[seed]:.0f} inhabitants")


if __name__ == "__main__":
    main()
