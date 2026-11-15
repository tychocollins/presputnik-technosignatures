# Phase 1 Feasibility Report: Pre-Sputnik Earth-Shadow Test

## Executive Verdict: **GO** (Re-evaluated with Hard Detectability Cut `m_flash(0.5s) ≥ 9.0`)

- **Tier A (`m_flash ≥ 9.0`) vs POSS-I-E raw shadow exposure:** **4.21×** (1,751.2 vs 415.6 deg²·exp)

- **Tier A (`m_flash ≥ 9.0`) fully-in-shadow exposures:** **40** across **7** independent instruments (thresholds: ≥ 1.0× POSS-I-E and ≥ 30 fully-in-shadow across ≥ 3 instruments).

- **Contaminated DASCH series removed:** Dropped 13 entire DASCH series (`ak`, `an`, `fa`, `hale`, `ka`, `kb`, `kc`, `kd`, `ke`, `kf`, `kh`, `lwla`, `mb`) with >10% Sun-check or pointing-check failure rates (**14,657** exposures removed, including 11,134 in Tier A and 5,630 `any_in_shadow`).

- **VASCO transient flash-magnitude range:** POSS-I-E limit ~20.0 mag with ~45 min exposures (`m_flash(0.5s) = 10.67` mag median); transients within 2 mag of plate limit (`18–20` mag) correspond to **`m_flash(0.5s) ≈ 8.67–10.67` mag**.


## 1. Detectability-Cut Analysis (`m_flash(0.5s) ≥ threshold`)

| m_flash_min | group | n_exp | n_any_shadow | n_fully_shadow | n_inst_fully | raw_shadow_deg2 |
| --- | --- | --- | --- | --- | --- | --- |
| 8.00 | POSS-I-E | 914 | 32 | 2 | 1 | 415.60 |
| 8.00 | Tier A | 19,727 | 381 | 60 | 8 | 10530.23 |
| 8.00 | Tier B | 46,319 | 1,128 | 157 | 10 | 29020.23 |
| 9.00 | POSS-I-E | 914 | 32 | 2 | 1 | 415.60 |
| 9.00 | Tier A | 7,593 | 115 | 40 | 7 | 1751.22 |
| 9.00 | Tier B | 17,683 | 396 | 86 | 7 | 6213.04 |
| 10.00 | POSS-I-E | 914 | 32 | 2 | 1 | 415.60 |
| 10.00 | Tier A | 1,727 | 56 | 20 | 7 | 811.44 |
| 10.00 | Tier B | 2,793 | 83 | 22 | 7 | 1439.05 |
| 10.67 | POSS-I-E | 108 | 3 | 1 | 1 | 53.63 |
| 10.67 | Tier A | 830 | 26 | 8 | 4 | 213.29 |
| 10.67 | Tier B | 1,229 | 35 | 9 | 4 | 282.17 |


## 2. Tier A Composition by Archive & Instrument (Sorted by `m_flash ≥ 9.0` Shadow Coverage)

| group | n_exposures | n_exp_m9 | n_any_in_shadow | n_fully_in_shadow | median_m_flash_05s | raw_shadow_m9_deg2 | raw_shadow_eff_deg2 | sens_shadow_eff_deg2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DASCH: 16-inch Metcalf Doublet (Refigured after 3500) | 6,441 | 3,947 | 118 | 17 | 9.80 | 545.99 | 1971.45 | 396.08 |
| DASCH: 8-inch Draper Doublet | 4,459 | 129 | 169 | 0 | 7.36 | 398.60 | 6216.37 | 593.55 |
| DASCH: 10-inch Metcalf Triplet | 13,669 | 295 | 752 | 13 | 7.17 | 236.99 | 23669.32 | 1356.90 |
| DASCH: 24-inch Bruce Doublet | 2,361 | 946 | 60 | 19 | 8.61 | 211.88 | 531.00 | 156.70 |
| DASCH: 32-36 inch BakerSchmidt 10 1/2 inch round Armagh-Dunsink-Harvard | 147 | 140 | 15 | 4 | 10.55 | 108.25 | 108.25 | 98.16 |
| DASCH: 24-33 in Jewett Schmidt | 1,176 | 881 | 9 | 3 | 9.80 | 96.94 | 97.33 | 68.49 |
| DASCH: 24-inch Clark Reflector | 652 | 356 | 19 | 17 | 9.26 | 54.49 | 55.55 | 30.47 |
| APPLAUSE: Original Schmidt-Spiegel | 652 | 31 | 7 | 0 | 5.27 | 41.78 | 131.63 | 29.41 |
| DASCH: 8-inch Bache Doublet | 6,902 | 192 | 322 | 15 | 7.80 | 27.64 | 14574.89 | 958.76 |
| DASCH: 13-inch Boyden Refractor | 913 | 83 | 27 | 10 | 8.80 | 15.81 | 112.45 | 18.25 |
| DASCH: 11-inch Draper Refractor | 2,856 | 81 | 8 | 3 | 7.92 | 12.84 | 37.42 | 13.73 |
| DASCH: 3 inch Ross-Tessar Lens | 668 | 0 | 303 | 0 | 5.11 | 0.00 | 46581.40 | 255.98 |
| DASCH: 2.5 inch Cooke Lens | 99 | 0 | 7 | 0 | 3.74 | 0.00 | 1015.44 | 1.57 |
| DASCH: 2.5 inch Voigtlander (Little Bache or "Bachito") | 12 | 0 | 3 | 0 | 2.43 | 0.00 | 468.50 | 0.22 |
| DASCH: 2.6-inch Zeiss-Tessar | 146 | 0 | 17 | 0 | 3.84 | 0.00 | 3031.63 | 5.28 |
| DASCH: 3 inch Perkin-Zeiss Lens | 17 | 0 | 7 | 0 | 2.85 | 0.00 | 1301.15 | 0.96 |
| APPLAUSE:  | 237 | 0 | 15 | 1 | 3.53 | 0.00 | 908.47 | 1.83 |
| DASCH: 3-inch Elmer Ross | 24 | 0 | 0 | 0 | 4.36 | 0.00 | 0.00 | 0.00 |
| DASCH: 3-inch Ross | 182 | 0 | 14 | 0 | 4.36 | 0.00 | 1031.07 | 2.95 |
| DASCH: 4-inch Cooke Lens | 816 | 0 | 10 | 0 | 6.80 | 0.00 | 760.44 | 6.78 |


## 3. Timing Perturbation Check (±30 min, 20 random draws)

| source_archive | n_sampled | base_in_shadow_pct | mean_flip_pct |
| --- | --- | --- | --- |
| APPLAUSE | 10,000 | 4.370 | 0.098 |
| DASCH | 10,000 | 11.300 | 0.228 |
| POSS-I | 914 | 3.501 | 0.246 |


## 4. Filter Loss Summary by Archive

| source_archive | sub_archive | raw_rows | pass_valid_time | pass_pre_sputnik | pass_wcs | pass_obs_coords | pass_sun_alt | pass_pointing_alt | drop_time | drop_post_sputnik | drop_no_wcs | drop_no_obs | drop_sun_alt | drop_pointing_alt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DASCH | DASCH-narrow | 337,254 | 181,989 | 181,989 | 181,928 | 181,928 | 178,739 | 177,378 | 155,265 | 0 | 61 | 0 | 3,189 | 1,361 |
| DASCH | DASCH-wide | 199,109 | 160,480 | 160,480 | 160,454 | 160,454 | 156,103 | 155,809 | 38,629 | 0 | 26 | 0 | 4,351 | 294 |
| APPLAUSE | Lippert-Astrograph (Hamburg) | 15,052 | 13,723 | 13,688 | 10,139 | 10,139 | 10,038 | 10,035 | 1,329 | 35 | 3,549 | 0 | 101 | 3 |
| APPLAUSE | Astrograph, Bamberg Northern Sky Patrol | 13,227 | 13,044 | 13,009 | 6,846 | 6,846 | 6,236 | 5,611 | 183 | 35 | 6,163 | 0 | 610 | 625 |
| APPLAUSE | 1m-Spiegelteleskop (Hamburg) | 10,924 | 8,308 | 8,306 | 5,638 | 5,638 | 5,622 | 5,593 | 2,616 | 2 | 2,668 | 0 | 16 | 29 |
| APPLAUSE | AG-Teleskop (Hamburg) | 4,336 | 3,795 | 3,647 | 3,533 | 3,533 | 3,453 | 3,452 | 541 | 148 | 114 | 0 | 80 | 1 |
| APPLAUSE | Zeiss Triplet (Potsdam) | 12,140 | 5,067 | 5,067 | 2,371 | 2,371 | 2,177 | 2,170 | 7,073 | 0 | 2,696 | 0 | 194 | 7 |
| APPLAUSE | Doppel-Reflektor (Hamburg) | 2,759 | 2,283 | 2,283 | 1,828 | 1,828 | 1,825 | 1,822 | 476 | 0 | 455 | 0 | 3 | 3 |
| APPLAUSE | Tartu Old Observatory | 1,687 | 1,652 | 1,626 | 1,604 | 1,604 | 1,600 | 1,600 | 35 | 26 | 22 | 0 | 4 | 0 |
| APPLAUSE | AG-Teleskop (Bonn) | 1,488 | 1,446 | 1,446 | 1,322 | 1,322 | 1,307 | 1,305 | 42 | 0 | 124 | 0 | 15 | 2 |
| APPLAUSE | Schmidtsches Spiegelteleskop (Hamburg) | 4,903 | 2,472 | 2,466 | 999 | 999 | 962 | 931 | 2,431 | 6 | 1,467 | 0 | 37 | 31 |
| POSS-I | POSS-I-E | 1,037 | 937 | 916 | 916 | 916 | 915 | 914 | 100 | 21 | 0 | 0 | 1 | 1 |
| APPLAUSE | Grosser Refraktor (Hamburg) | 8,865 | 2,864 | 2,864 | 769 | 769 | 765 | 741 | 6,001 | 0 | 2,095 | 0 | 4 | 24 |
| APPLAUSE | Grosser Schmidt-Spiegel (Hamburg) | 5,453 | 1,399 | 1,336 | 512 | 512 | 506 | 500 | 4,054 | 63 | 824 | 0 | 6 | 6 |
| APPLAUSE | Zeiss Double Astrograph | 543 | 543 | 543 | 530 | 530 | 272 | 176 | 0 | 0 | 13 | 0 | 258 | 96 |
| APPLAUSE | Kleiner Schmidt-Spiegel II (Hamburg) | 3,337 | 232 | 222 | 89 | 89 | 87 | 84 | 3,105 | 10 | 133 | 0 | 2 | 3 |
| APPLAUSE | Great Schmidt Camera (Potsdam) | 718 | 39 | 39 | 29 | 29 | 29 | 28 | 679 | 0 | 10 | 0 | 0 | 1 |
| APPLAUSE | Ross Camera (Potsdam) | 83 | 64 | 64 | 59 | 59 | 17 | 17 | 19 | 0 | 5 | 0 | 42 | 0 |
| APPLAUSE | Small Schmidt Camera (Potsdam) | 494 | 66 | 66 | 13 | 13 | 5 | 5 | 428 | 0 | 53 | 0 | 8 | 0 |
| APPLAUSE | Hamburger Schmidt-Spiegel (Calar Alto) | 1,912 | 0 | 0 | 0 | 0 | 0 | 0 | 1,912 | 0 | 0 | 0 | 0 | 0 |
| APPLAUSE | Tautenburg Schmidt Telescope | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 |
| APPLAUSE | ESO telescopes (La Silla) | 632 | 0 | 0 | 0 | 0 | 0 | 0 | 632 | 0 | 0 | 0 | 0 | 0 |
| APPLAUSE | Carte du Ciel (Potsdam) | 1,589 | 979 | 979 | 0 | 0 | 0 | 0 | 610 | 0 | 979 | 0 | 0 | 0 |
| APPLAUSE | Bamberg Southern Sky Patrol (Argentina) | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
