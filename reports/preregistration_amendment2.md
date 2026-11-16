# Pre-Registration, Amendment 2
### SUBTITLE: Co-primary adjustment for realized quality-check sample sizes before shadow-count unblinding
### SUBTITLE: Tycho Collins (independent researcher)
### SUBTITLE: Frozen 16 November 2026, before any in-shadow or out-of-shadow candidate counts were computed

---

## 1. Summary and reason for Amendment 2

Following OSF registration of Amendment 1 (DOI: 10.17605/OSF.IO/5PQDZ; project: https://osf.io/h7rqn), all 1,117 Phase 2b digitized plates (DASCH and APPLAUSE) were extracted using the frozen pipeline (src/phase2b_extract.py, zero parameter changes) and evaluated at **Step 1 (quality-check fairness by instrument)** before any candidate shadow positions or in-shadow versus out-of-shadow counts were computed.

**Step 1 passed overall quality-check fairness:**
- Against local-solar-time-matched sunlit ecliptic midnight plates at flash magnitude ≥ 9.0 (shadow fraction = 0), Stage A partially shadowed plates passed quality checks at **22.1% (61 / 276)** versus **22.2% expected (61.2 / 276; raw 23 / 100 = 23.0%)**, stratified Cochran–Mantel–Haenszel odds ratio = 0.970, two-sided p = 0.9579.
- Across all partially shadowed plates at flash magnitude ≥ 8.0 (810 unique plates), quality-check pass was **36.2% (293 / 810)** versus **36.3% expected (294.2 / 810)** from sunlit plates, stratified odds ratio = 0.962, two-sided p = 0.7765.
- Two southern Boyden instruments differed from all-sky sunlit rates and are logged: **Baker-Schmidt (ADH)** passed 3 / 9 (33.3%) versus 31 / 34 (91.2%, Fisher p = 0.0009, surviving Bonferroni across 13 instruments; 18 / 20 = 90.0%, p = 0.0039 against near-ecliptic sunlit fields where the southern ecliptic crosses the dense Sagittarius bulge; 3 / 3, p = 0.1818 against matched midnight plates; 4 of the 6 failures come from one anomalous pre-first-light logbook night, MJD 31614, leaving 7 / 11 = 63.6%, p = 0.149 across all other nights; contributes 3 plates to the mag ≥ 9.0 sample), and **8-inch Bache** passed 0 / 3 at mag ≥ 9.0 and 30 / 60 (50.0%) at mag ≥ 8.0 versus 80.0% / 72.0% all-sky sunlit (p = 0.0122, does not survive Bonferroni; disappears against near-ecliptic sunlit fields at 9 / 15 = 60.0%, p = 0.5706).

**However, Step 1 revealed a pre-outcome sample-size projection error in the mag ≥ 9.0 test:**
- Amendment 1 projected N = 106 passing plates at flash magnitude ≥ 9.0 by multiplying the 133 16-inch Metcalf exposures at mag ≥ 9.0 by Metcalf's *pooled mag ≥ 8.0* pass rate (39.5%, driven by long-exposure mag 8.0–9.0 plates at 68.9%) rather than Metcalf's short-exposure (600 s) *mag ≥ 9.0* pass rate (15.0% in shadow, 14.6% in matched sunlit midnight, p = 1.00).
- As a result, the realized mag ≥ 9.0 sample has **N = 61 passing plates** instead of 106, reducing its realistic depth-scaled power at a 30% deficit from 77.9% to **59.2%** (80.2% unscaled).
- Meanwhile, the mag ≥ 8.0 partially shadowed sample has **N = 293 passing plates** (93.3% of the 314 projected), with both pre-registered safeguards (time-shifted 3D ecliptic placebos and 32 × 32 Gaia star-density control) already frozen in place, and the strict paired Tier A check grew from 66 projected to **N = 95 passing target exposures**.

## 2. Blinding guarantee at Amendment 2

**Zero candidate shadow classifications or in-shadow versus out-of-shadow counts have been computed on any shadow plate.** Only plate-level quality-check pass flags (qc_pass), photometric band tags (phot_band), instrument labels, and Phase 1 archive shadow-fraction metadata were read in Step 1 and in src/amendment2_power_calibration.py. All candidate counts, spatial defect profiles, Gaia density grids, and placebo distributions in Section 4 come 100% from the strictly sunlit (shadow_fraction = 0 at 42,164, 36,000 and 20,000 km) calibration pool (pilot + sunlit_m8, 773 sunlit exposures). Steps 2 through 5 remain unexecuted until Amendment 2 is timestamped on OSF.

## 3. Amendment 2 change: Holm co-primary test

**Nothing in extraction, calibration, quality checks, Final Frozen Filter v1.2, expected in-shadow share p_i, negative controls, or the strict paired Tier A directional check changes.** The sole change is promoting the single-plate within-plate test at flash magnitude ≥ 8.0 (realized N = 293) from secondary to **co-primary** alongside flash magnitude ≥ 9.0 (realized N = 61), using **Holm's step-down procedure** to control the one-sided familywise false-positive rate at α = 0.05:

1. **Step 1 of Holm procedure (α = 0.025):** Test the mag ≥ 8.0 single-plate within-plate test (N = 293) against at least 1,000 (specifically 2,000) time-shifted 3D ecliptic shadow placebos at one-sided **p_placebo ≤ 0.025**.
2. **Step 2 of Holm procedure (α = 0.050):** If mag ≥ 8.0 is significant at p_placebo ≤ 0.025, test the mag ≥ 9.0 single-plate within-plate test (N = 61) at one-sided **p_placebo ≤ 0.050** *(under symmetric two-test Holm, if mag ≥ 8.0 exceeds 0.025, mag ≥ 9.0 requires p_placebo ≤ 0.025; coupled sunlit simulations verify familywise error rate = 2.3% fixed-order / 4.3% symmetric Holm, both ≤ 5.0%)*.

### Updated Decision Rule (Three Gates — Unchanged in Structure)

A shadow deficit will be reported as positive evidence for a pre-Sputnik population of orbital flashes **only if all three gates hold simultaneously**:
- **(a) Co-primary significance (Holm-adjusted, FWER ≤ 0.05):** Test mag ≥ 8.0 (N = 293) at α = 0.025; if significant, test mag ≥ 9.0 (N = 61) at α = 0.050 (under symmetric two-test Holm, either co-primary passing at p_placebo ≤ 0.025 satisfies Gate a).
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
| **Secondary: full-shadow across-plate, mag ≥ 9.0 (N = 47)** | **0.050** | **-1.723** | **5.2%** | 7.6% / 9.7% | **9.6% / 11.2%** | **10.9% / 16.5%** | **16.4% / 19.1%** |

**Additional notes on the realized calibration:**
- **Co-Primary 1 (mag ≥ 8.0, N = 293) at α = 0.025:** Exceeds the 80% power benchmark at a 30% deficit (**83.0%** depth-scaled under the frozen sunlit bank's depth mix; **98.3%** unscaled). Under an even harsher coupled sensitivity check that evaluates the linear depth-dilution scale (1.0 at mag ≥ 10.67 down to 0.0 at mag 7.0) at each realized plate's exact archive flash magnitude (79.2% in [8.0, 9.0), mean depth factor 0.471 × 0.531 star share = 0.250 net multiplier), unscaled power at α = 0.025 is **99.6%** at a 30% deficit and depth-scaled Holm power is **57.4%** fixed-order / **67.4%** symmetric Holm at 30% (**85.7% / 91.3%** at 40%).
- **Strict Paired Tier A (mag ≥ 8.0, N = 95):** Standalone α = 0.05 power at 30% rose from 19.4% / 22.3% (at projected N = 66) to **35.8% / 49.8%** (reaching **89.5% / 99.2%** at 60% and **97.2% / 99.9%** at 70%), while its directional concordance probability (RR_MH < 1 and Z < 0, Gate c) is **94.9% / 98.3%** at a 30% deficit.
- **Full-Shadow Dose-Response (mag ≥ 9.0, shadow fraction ≥ 0.95, N = 47 vs. 231 sunlit mag ≥ 9.0 plates):** Across-plate comparisons remain limited by between-plate count overdispersion (9.6% / 11.2% standalone power at 30%), underscoring why within-plate conditioning is used for the co-primary tests.

## Appendix. Frozen commits and Amendment 2 files

- **Repository:** https://github.com/tychocollins/presputnik-technosignatures
- **Prior OSF Registration (Amendment 1):** https://doi.org/10.17605/OSF.IO/5PQDZ (584dda36f9441c9b83b0f06ad0cbf9ee1eb5f655)
- **Frozen Extraction & Filter Code Commit (Unchanged):** a41c00421f968b1c0c46f75c7c765b4406da72c0
- **Amendment 2 Realized-Sample Power Calibration Commit:** 1f1260807698250a28055989296472c9d03af131
- **Amendment 2 Calibration & Step 1 QC Files:**
  - src/amendment2_power_calibration.py
  - data/processed/amendment2_realized_power_calibration.csv
  - data/processed/phase2b_step1_qc_fairness.csv
  - reports/methods_log.md
