N_dis = 10**2   # 捨てる個数
N_trj = 10**3   # トータルのアップデート回数
N_sav = 10      # アップデート10回ごとに配位を保存
prm_list = [
    # beta, # of_trj, # of_discard, file_name, save_every
    [0.90, N_trj, N_dis, "data/conf/L32b090_", N_sav],
    [0.85, N_trj, N_dis, "data/conf/L32b085_", N_sav],
    [0.80, N_trj, N_dis, "data/conf/L32b080_", N_sav],
    [0.75, N_trj, N_dis, "data/conf/L32b075_", N_sav],
    [0.70, N_trj, N_dis, "data/conf/L32b070_", N_sav],
    [0.65, N_trj, N_dis, "data/conf/L32b065_", N_sav],
    [0.60, N_trj, N_dis, "data/conf/L32b060_", N_sav],
    [0.55, N_trj, N_dis, "data/conf/L32b055_", N_sav],
    [0.50, N_trj, N_dis, "data/conf/L32b050_", N_sav],
    [0.47, N_trj, N_dis, "data/conf/L32b047_", N_sav],
    [0.42, N_trj, N_dis, "data/conf/L32b042_", N_sav],
    [0.40, N_trj, N_dis, "data/conf/L32b040_", N_sav],
    [0.35, N_trj, N_dis, "data/conf/L32b035_", N_sav],
    [0.30, N_trj, N_dis, "data/conf/L32b030_", N_sav],
    [0.25, N_trj, N_dis, "data/conf/L32b025_", N_sav],
    [0.20, N_trj, N_dis, "data/conf/L32b020_", N_sav],
    [0.15, N_trj, N_dis, "data/conf/L32b015_", N_sav],
    [0.10, N_trj, N_dis, "data/conf/L32b010_", N_sav],
    [0.05, N_trj, N_dis, "data/conf/L32b005_", N_sav],
    [0.00, N_trj, N_dis, "data/conf/L32b000_", N_sav],
    ]

prm_list_tri = [
    # beta, # of_trj, # of_discard, file_name, save_every
    # [1.00, N_trj, N_dis, "data/conf_tri/L32b025_", N_sav],  
    [0.54, N_trj, N_dis, "data/conf_tri/L32b090_", N_sav],  
    [0.51, N_trj, N_dis, "data/conf_tri/L32b090_", N_sav],
    [0.48, N_trj, N_dis, "data/conf_tri/L32b085_", N_sav],
    [0.45, N_trj, N_dis, "data/conf_tri/L32b080_", N_sav],
    [0.42, N_trj, N_dis, "data/conf_tri/L32b075_", N_sav],
    [0.39, N_trj, N_dis, "data/conf_tri/L32b070_", N_sav],
    [0.36, N_trj, N_dis, "data/conf_tri/L32b065_", N_sav],
    [0.33, N_trj, N_dis, "data/conf_tri/L32b060_", N_sav],
    [0.31, N_trj, N_dis, "data/conf_tri/L32b055_", N_sav],
    [0.29, N_trj, N_dis, "data/conf_tri/L32b050_", N_sav],
    [0.25, N_trj, N_dis, "data/conf_tri/L32b047_", N_sav],
    [0.23, N_trj, N_dis, "data/conf_tri/L32b040_", N_sav],
    [0.21, N_trj, N_dis, "data/conf_tri/L32b035_", N_sav],
    [0.18, N_trj, N_dis, "data/conf_tri/L32b030_", N_sav],
    [0.15, N_trj, N_dis, "data/conf_tri/L32b025_", N_sav],
    [0.12, N_trj, N_dis, "data/conf_tri/L32b020_", N_sav],
    [0.09, N_trj, N_dis, "data/conf_tri/L32b015_", N_sav],
    [0.06, N_trj, N_dis, "data/conf_tri/L32b010_", N_sav],
    [0.03, N_trj, N_dis, "data/conf_tri/L32b005_", N_sav],
    [0.00, N_trj, N_dis, "data/conf_tri/L32b000_", N_sav],
    ]