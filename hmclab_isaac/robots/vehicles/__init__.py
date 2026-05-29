"""Lab vehicles. Each subpackage owns one vehicle family.

Currently supported:
    HAMA       — DBXL chassis + Mid-360 LiDAR (variants HAMA_1, HAMA_2)
    UNICORN    — SRC chassis + Mid-360 LiDAR (variants UNICORN_1, UNICORN_2)
    ROBORACER  — placeholder

Per-vehicle layout:
    <FAMILY>/
        <FAMILY>.py        single config module: ArticulationCfg, sensor
                            factory, geometry constants, variant lookup.
        <FAMILY>_1.usd     assembly USD that references the bare chassis
                            from `../../chassis/` and devices from
                            `../../devices/`. No geometry is duplicated.
        <FAMILY>_N.usd     additional chassis × sensor combos.
"""
