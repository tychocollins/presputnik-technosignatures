# Pre-Registration, Amendment 2
### SUBTITLE: Co-primary adjustment and instrument date-sanity rule before shadow-count unblinding
### SUBTITLE: Tycho Collins (independent researcher)
### SUBTITLE: Frozen 16 November 2026, before any in-shadow or out-of-shadow candidate counts were computed

---

## 1. Summary and reasons for Amendment 2

Following OSF registration of Amendment 1 (DOI: 10.17605/OSF.IO/5PQDZ; project: https://osf.io/h7rqn), all 1,117 Phase 2b digitized plates (DASCH and APPLAUSE) were extracted using the frozen pipeline (src/phase2b_extract.py, zero parameter changes) and evaluated at **Step 1 (quality-check fairness by instrument)** before any candidate shadow positions or in-shadow versus out-of-shadow counts were computed.

**Step 1 passed overall quality-check fairness:**
- Against local-solar-time-matched sunlit ecliptic midnight plates at flash magnitude ≥ 9.0 (shadow fraction = 0), Stage A partially shadowed plates passed quality checks at **22.1% (61 / 276)** versus **22.2% expected (61.2 / 276; raw 23 / 100 = 23.0%)**, stratified Cochran–Mantel–Haenszel odds ratio = 0.970, two-sided p = 0.9579.
- Across all partially shadowed plates at flash magnitude ≥ 8.0 (810 unique plates), quality-check pass was **36.2% (293 / 810)** versus **36.3% expected (294.2 / 810)** from sunlit plates, stratified odds ratio = 0.962, two-sided p = 0.7765.
- Two southern Boyden instruments differed from all-sky sunlit rates and are logged: **Baker-Schmidt (ADH)** passed 3 / 9 (33.3%) versus 31 / 34 (91.2%, Fisher p = 0.0009, surviving Bonferroni across 13 instruments; 18 / 20 = 90.0%, p = 0.0039 against near-ecliptic sunlit fields where the southern ecliptic crosses the dense Sagittarius bulge; 3 / 3, p = 0.1818 against matched midnight plates; 4 of the 6 failures are misdated 1945-06-08/09, five years before ADH first light in Nov 1950, leaving 7 / 11 = 63.6%, p = 0.149 across valid nights; contributes 3 plates to the mag ≥ 9.0 sample), and **8-inch Bache** passed 0 / 3 at mag ≥ 9.0 and 30 / 60 (50.0%) at mag ≥ 8.0 versus 80.0% / 72.0% all-sky sunlit (p = 0.0122, does not survive Bonferroni; disappears against near-ecliptic sunlit fields at 9 / 15 = 60.0%, p = 0.5706).

**Step 1 surfaced two pre-outcome issues addressed in Amendment 2 before any shadow counts are computed:**
1. **Sample-size projection error on the mag ≥ 9.0 test:** Amendment 1 projected N = 106 passing plates at flash magnitude ≥ 9.0 by multiplying the 133 16-inch Metcalf exposures at mag ≥ 9.0 by Metcalf's *pooled mag ≥ 8.0* pass rate (39.5%, driven by long-exposure mag 8.0–9.0 plates at 68.9%) rather than Metcalf's short-exposure (600 s) *mag ≥ 9.0* pass rate (15.0% in shadow, 14.6% in matched sunlit midnight, p = 1.00). Realized N fell to **61 passing plates**, reducing depth-scaled power at a 30% deficit from 77.9% to **59.2%** (80.2% unscaled). Meanwhile, the mag ≥ 8.0 partially shadowed sample has **N = 293 passing plates** (93.3% of 314 projected) with both pre-registered safeguards (time-shifted 3D ecliptic placebos and 32 × 32 Gaia star-density control) frozen in place, and strict paired Tier A grew from 66 projected to **N = 95 passing target exposures**.
2. **Impossible logbook dates and shadow geometry:** Because Earth's umbral shadow calculation depends directly on exposure date, the four ADH plates dated 1945 (before the telescope existed) motivated a systematic instrument operational date-sanity check across all archives and telescopes.

## 2. Blinding guarantee at Amendment 2

**Zero candidate shadow classifications or in-shadow versus out-of-shadow counts have been computed on any shadow plate.** Only plate-level quality-check pass flags (qc_pass), photometric band tags (phot_band), instrument labels, exposure timestamps (mjd_mid), and Phase 1 archive shadow-fraction metadata were read in Step 1 and in src/amendment2_power_calibration.py. All candidate counts, spatial defect profiles, Gaia density grids, and placebo distributions in Section 4 come 100% from the strictly sunlit (shadow_fraction = 0 at 42,164, 36,000 and 20,000 km) calibration pool (pilot + sunlit_m8, 773 sunlit exposures). Steps 2 through 5 remain unexecuted until Amendment 2 is timestamped on OSF.

## 3. Amendment 2 changes

**All extraction, calibration, quality-check, Final Frozen Filter v1.2, expected in-shadow share p_i, negative-control, and paired Tier A partner rules remain unchanged.** Three pre-outcome rules are added:

### 3.1 Frozen instrument operational date-sanity rule
Any exposure whose mid-exposure UTC date falls outside its instrument's approximate operational bounds (Appendix A; implemented in filter_instrument_date_sanity in src/amendment2_power_calibration.py) is excluded prior to analysis. Applying this rule across all 1,117 Phase 2b exposures and all 773 sunlit calibration exposures removes **4 raw QC-failing exposures** (DASCH_adh02437, adh02442, adh02443, adh02448, dated 1945-06-08/09, all qc_pass = False) and **0 QC-passing plates** from every test:
- Co-Primary 1 (mag ≥ 8.0 single-plate): **0 QC-passing plates removed** (N = 293 unchanged; 293 / 806 = 36.4% valid raw).
- Co-Primary 2 (mag ≥ 9.0 single-plate): **0 QC-passing plates removed** (N = 61 unchanged; 61 / 272 = 22.4% valid raw).
- Key Secondary (strict paired Tier A, mag ≥ 8.0): **0 QC-passing target or partner plates removed** (N = 95 unchanged).
- Secondary full-shadow across-plate (mag ≥ 9.0, shadow fraction ≥ 0.95): **0 QC-passing plates removed** (N = 47 unchanged).
- Matched-LST sunlit midnight control (N = 27) and 773-exposure sunlit calibration pool: **0 plates removed**.

Because N is unchanged across every test, no further power adjustment is required.

### 3.2 Co-primary multiplicity rule: fixed-sequence, Holm-type step-down
The single-plate within-plate test at flash magnitude ≥ 8.0 (realized N = 293) becomes co-primary alongside flash magnitude ≥ 9.0 (realized N = 61) using a **fixed-sequence, Holm-type step-down (m ≥ 8.0 first at α = 0.025; m ≥ 9.0 at α = 0.05 only if the first is significant)**:
1. **Step 1 (α = 0.025):** Test mag ≥ 8.0 single-plate within-plate (N = 293) against at least 1,000 (specifically 2,000) time-shifted 3D ecliptic shadow placebos at one-sided **p_placebo ≤ 0.025**. If p_placebo > 0.025, stop: neither co-primary is declared significant.
2. **Step 2 (α = 0.050, conditional on Step 1):** Only if mag ≥ 8.0 is significant at p_placebo ≤ 0.025, test mag ≥ 9.0 single-plate within-plate (N = 61) at one-sided **p_placebo ≤ 0.050** *(coupled sunlit simulations verify one-sided familywise error rate = 2.3% ≤ 5.0%)*.

### 3.3 Updated Decision Rule (Three Gates — Unchanged in Structure)
A shadow deficit will be reported as positive evidence for a pre-Sputnik population of orbital flashes **only if all three gates hold simultaneously**:
- **(a) Co-primary significance (fixed-sequence, Holm-type step-down, FWER ≤ 0.05):** The mag ≥ 8.0 single-plate test (N = 293) is significant at one-sided p_placebo ≤ 0.025 (and if so, mag ≥ 9.0, N = 61, is tested at p_placebo ≤ 0.050).
- **(b) All three negative controls pass:** (1) shifted dates (±1 to ±5 days and ±30, ±60, ±90 days give the strongest deficit near zero shift and no significant deficit, p > 0.05, away from it), (2) fake altitudes (42,164 km vs. 36,000, 20,000 and 5,000 km behave as expected for objects at high altitude rather than a fixed pattern on the plate), and (3) same local time, no shadow (sunlit midnight plates, realized N = 27 passing plates at mag ≥ 9.0, shadow fraction = 0, ecliptic latitude 5° to 18°, show no deficit).
- **(c) Strict paired Tier A points the same way:** On the realized N = 95 passing rapid-repeat emulsion- and band-matched target exposures (mag ≥ 8.0), Mantel–Haenszel rate ratio is below 1 and Z < 0 (standalone p ≤ 0.05 not required; simulated probability of satisfying RR_MH < 1 and Z < 0 under a true 30% deficit is **94.9%** depth-scaled / **98.3%** unscaled).

If (a) fails, the result is **null**. If (a) passes but (b) or (c) fails, the result is **inconclusive or caused by an artefact**.

## 4. Recalibrated power at realized sample sizes (2,000 sunlit simulations)

All figures below (data/processed/amendment2_realized_power_calibration.csv) are computed strictly from the 773-exposure sunlit pool (shadow_fraction = 0) at the realized Step 1 quality-passing sample sizes, allowing for the measured 0.531 non-stellar share on single plates. Power cells show **depth-scaled / unscaled** values across 20%, 30%, 40%, and 50% deficits:

| Test (Realized QC Plates) | α | Crit. z | Placebo FPR | Power @ 20% | Power @ 30% | Power @ 40% | Power @ 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Co-Primary 1: single plate, mag ≥ 8.0 (N = 293, Gaia)** | **0.025** | **-2.669** | **2.2%** | 47.6% / 76.9% | **83.0% / 98.3%** | **97.1% / 100%** | **100% / 100%** |
| Co-Primary 1: single plate, mag ≥ 8.0 (N = 293, Gaia) | 0.050 | -2.116 | 4.7% | 60.0% / 86.1% | 90.6% / 99.4% | 98.9% / 100% | 100% / 100% |
| Ablation: single plate, mag ≥ 8.0 (N = 293, no Gaia) | 0.025 | -2.607 | 2.6% | 49.7% / 79.5% | 82.6% / 98.5% | 97.1% / 100% | 99.9% / 100% |
| **Co-Primary 2: single plate, mag ≥ 9.0 (N = 61, Gaia)** | **0.050** | **-2.428** | **4.9%** | 33.5% / 52.1% | **59.2% / 80.2%** | **79.5% / 95.7%** | **92.1% / 99.4%** |
| Co-Primary 2: single plate, mag ≥ 9.0 (N = 61, Gaia) | 0.025 | -2.991 | 2.4% | 22.9% / 37.0% | 46.2% / 69.2% | 68.0% / 91.7% | 86.2% / 98.5% |
| Ablation: single plate, mag ≥ 9.0 (N = 61, no Gaia) | 0.050 | -2.312 | 6.0% | 36.9% / 53.5% | 61.3% / 82.9% | 82.2% / 96.5% | 92.5% / 99.8% |
| **Key Secondary: strict paired Tier A, mag ≥ 8.0 (N = 95)** | **0.050** | **-1.962** | **4.6%** | 22.0% / 28.4% | **35.8% / 49.8%** | **58.6% / 78.5%** | **78.8% / 93.3%** |
| **Uninformative (pre-reg.): full-shadow, mag ≥ 9.0 (N = 47)** | **0.050** | **-1.723** | **5.2%** | 7.6% / 9.7% | **9.6% / 11.2%** | **10.9% / 16.5%** | **16.4% / 19.1%** |

**Notes on the realized calibration:**
- **Co-Primary 1 (mag ≥ 8.0, N = 293) at α = 0.025:** Exceeds the 80% power benchmark at a 30% deficit (**83.0%** depth-scaled under the frozen sunlit bank's depth mix; **98.3%** unscaled). Under an even harsher coupled sensitivity check evaluating the linear depth-dilution scale (1.0 at mag ≥ 10.67 down to 0.0 at mag 7.0) at each realized plate's exact archive flash magnitude (79.2% in [8.0, 9.0), mean depth factor 0.471 × 0.531 star share = 0.250 net multiplier), unscaled power at α = 0.025 is **99.6%** at 30% and fixed-sequence depth-scaled power is **57.4%** at 30% (**85.7%** at 40%; joint probability of rejecting both mag ≥ 8.0 at 0.025 and mag ≥ 9.0 at 0.05 is 42.5% / 79.8% at 30% and 74.1% / 96.4% at 40%).
- **Strict Paired Tier A (mag ≥ 8.0, N = 95):** Standalone α = 0.05 power at 30% rose from 19.4% / 22.3% (at projected N = 66) to **35.8% / 49.8%** (**89.5% / 99.2%** at 60%, **97.2% / 99.9%** at 70%), while its directional gate probability (RR_MH < 1 and Z < 0, Gate c) is **94.9% / 98.3%** at a 30% deficit.
- **Full-Shadow Across-Plate Test (mag ≥ 9.0, shadow fraction ≥ 0.95, N = 47 vs. 231 sunlit mag ≥ 9.0 plates — Pre-Registered but Uninformative):** Because plate-to-plate candidate count overdispersion overwhelms a 47-plate across-plate comparison, standalone power is **≈10% (9.6% depth-scaled / 11.2% unscaled at 30%)**. Because it was pre-registered, it will be retained and reported for completeness, **explicitly labeled uninformative and without substantive interpretation**.

## Appendix A. Approximate instrument operational date-sanity bounds

Exposures outside the approximate UTC operational bounds below (compiled from the Harvard plate series catalog and APPLAUSE DR4 metadata) are excluded prior to analysis. Verified sources:
- **[1] DASCH & Harvard Plate Stacks / StarGlass Series Catalog** (https://starglass.cfa.harvard.edu/): Laycock, S., Tang, S., Grindlay, J., Los, E., Simcoe, R., & Mink, D. (2010), "Digital Access to a Sky Century at Harvard. II: Initial Photometry and Astrometry," *AJ*, 140(4), 1062–1077 (ADS: 2010AJ....140.1062L); Grindlay, J., Tang, S., Los, E., & Servillat, M. (2012), "Opening the 100-Year Window for Time-Domain Astronomy," *IAU Symp.* 285, 29–34 (ADS: 2012IAUS..285...29G).
- **[2] ADH Baker-Schmidt & 24-inch Bruce Transition:** Butler, C. J. (2007), "The Armagh-Dunsink-Harvard Telescope: From Dream to Oblivion," *Journal of Astronomical History and Heritage*, 10(3), 173–178 (ADS: 2007JAHH...10..173B), documenting erection of the ADH Baker-Schmidt at Boyden Observatory in 1950 on the mounting of the retired 24-inch Bruce Doublet. Direct inspection of the raw DASCH adh catalog confirms plate #1 (adh00001) was exposed on 1950-11-27 (adh00002–00005 on 1950-11-30), whereas adh02437–02450 (dated 1945-06-08 to 1945-06-19) are a 1954 → 1945 digit transposition bracketed by adh02436 (1954-06-08) and adh02451 (1954-06-20).
- **[3] APPLAUSE DR4 Metadata & Instrument Catalog** (https://www.plate-archive.org/): Enke, H., Tuvikene, T., Groote, D., Edelmann, H., & Heber, U. (2024), "Archives of Photographic PLates for Astronomical USE (APPLAUSE): Digitisation of astronomical plates and their integration into the International Virtual Observatory," arXiv:2404.17355.

| Archive / Series | Instrument | Approximate Operational Bounds (UTC) | Source | Removed (Raw / QC) |
|---|---|---|---|---:|
| DASCH `mc` | 16-inch Metcalf Doublet (Refigured after 3500) | 1909-01-01 to 1989-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `mf` | 10-inch Metcalf Triplet | 1914-01-01 to 1956-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `ma` | 12-inch Metcalf Doublet | 1905-01-01 to 1953-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `a` | 24-inch Bruce Doublet | 1893-01-01 to 1950-11-01 (retired 1950 for ADH mount) | [1, 2] | 0 / 0 |
| DASCH `adh` | 32–36 in Baker-Schmidt (Armagh-Dunsink-Harvard) | 1950-11-01 to 1981-12-31 (adh00001 = 1950-11-27) | [1, 2] | **4 / 0** |
| DASCH `j` | 24–33 in Jewett Schmidt | 1941-01-01 to 1958-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `ir` / `rl` | 8-inch & 4-inch Ross-Lundin | 1934-01-01 to 1976-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `b` | 8-inch Bache Doublet | 1885-01-01 to 1955-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `i` / `c` | 8-inch Draper Doublet / 11-inch Draper Refractor | 1887-01-01 to 1946-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `x` | 13-inch Boyden Refractor | 1887-01-01 to 1950-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `h` | 24-inch Clark Reflector | 1904-01-01 to 1937-12-31 (approx., series catalog) | [1] | 0 / 0 |
| DASCH `ac`/`rh`/`br` | 1.5-in Cooke, 3-in Ross-Fecker, 8-in Brashear | Approx. series catalog bounds in code | [1] | 0 / 0 |
| APPLAUSE `103` | 1m-Spiegelteleskop (Hamburg Bergedorf) | 1911-01-01 to 1998-12-31 (approx., APPLAUSE metadata) | [3] | 0 / 0 |
| APPLAUSE `101`/`107`/`109` | Lippert-Astrograph, Doppel-Reflektor, Grosser Refraktor | Approx. APPLAUSE metadata bounds in code | [3] | 0 / 0 |
| APPLAUSE `111`/`110`/`102` | Original Schmidt, Kleiner Schmidt II, Grosser Schmidt | Approx. APPLAUSE metadata bounds in code | [3] | 0 / 0 |
| APPLAUSE `208`/`301` | Bamberg Northern Patrol Astrograph, Tartu 6-in Petzval | Approx. APPLAUSE metadata bounds in code | [3] | 0 / 0 |

## Appendix B. Frozen commits, Amendment 2 files, and Amendment 1 reference check

- **Repository:** https://github.com/tychocollins/presputnik-technosignatures
- **Prior OSF Registration (Amendment 1):** https://doi.org/10.17605/OSF.IO/5PQDZ (584dda36f9441c9b83b0f06ad0cbf9ee1eb5f655)
- **Errata / Reference Check on Amendment 1:** Audited preregistration_amendment1.pdf (and repository documentation; no README exists): Amendment 1 contains zero formal bibliographic citations, so no errata to Amendment 1 are required.
- **Frozen Extraction & Filter Code Commit (Unchanged):** a41c00421f968b1c0c46f75c7c765b4406da72c0
- **Amendment 2 Realized-Sample Calibration & Date-Sanity Commit:** 822aa05aab17570fec03b2c50fba7d3463f32091
- **Amendment 2 Files:** src/amendment2_power_calibration.py, data/processed/amendment2_realized_power_calibration.csv, data/processed/phase2b_step1_qc_fairness.csv, reports/methods_log.md
