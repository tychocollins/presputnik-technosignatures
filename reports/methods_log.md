# Project Alien: Methods Log

This log records the end-to-end methodology, exact parameters, and verified milestones across Phase 1 (Feasibility) and Phase 2 (Blind Pilot & Unblinded Shadow Test).

---

## Phase 1: Feasibility & Data Hygiene (Completed — GO for Blind Pilot)

1. **Archive Ingestion & Pre-Sputnik Filtering:** Collected pre-Sputnik (`< 1957-10-04T00:00:00 UTC`) plate exposure metadata from APPLAUSE DR4, Harvard DASCH (narrow & wide plates), and POSS-I-E (`VI/25/nposs` as the VASCO benchmark), totaling **368,171** raw exposures.
2. **Timestamp & Observatory Site Validation:** Verified `ObsE` is exposure start in PST (`UTC - 8h`, median offset vs `EHA` = `+0.08 min`). Mapped DASCH plate series to published observatory sites and enforced both a Sun-altitude check (`sun_alt_mid_deg < -12°`) and a pointing-altitude check (`pointing_alt_mid_deg > 10°`).
3. **Systematic Timing-Error Series Removal:** Dropped **13** entire DASCH series (`ak`, `an`, `fa`, `hale`, `ka`, `kb`, `kc`, `kd`, `ke`, `kf`, `kh`, `lwla`, `mb`) with `>10%` Sun- or pointing-check failure rates (**14,657** exposures removed), leaving **353,514** clean exposures.
4. **APPLAUSE Limiting Magnitude Correction:** Fixed APPLAUSE `m_lim` to read `phot_calib.faint_limit` rather than `plate_mag_lim`.
5. **Topocentric Earth-Shadow Geometry & Independent Verification:** Computed exact topocentric Earth-umbra shadow fractions at `42,164 km` (GEO), `36,000 km`, and `20,000 km`. Verified against an independent Skyfield (`de421.bsp`) implementation on 50 exposures (35 boundary exposures with `0.1 < shadow_fraction < 0.9` and 15 non-boundary exposures): `Median |diff| = 0.000000`, `Max |diff| = 0.001481`. Verified `±30 min` timing perturbation stability.
6. **Hard Detectability Cut (`m_flash(0.5s) ≥ 9.0`):** Replaced sensitivity weighting with a hard detectability cut `m_flash = m_lim - 2.5 * log10(exptime / 0.5s) >= 9.0`, comparable to our estimate of VASCO's flashes (derived by converting POSS-I-E transient plate magnitudes to 0.5 s flash equivalents).
7. **Both-Plates-Deep Pairing Rule:** Required **both** plates in every Tier A pair to satisfy `m_flash(0.5s) >= 9.0` so the partner plate provides a genuine defect check.
8. **Phase 1 Tier A Yield (`m_flash ≥ 9.0`, both plates deep):** **9,223** exposures, **173** with `any_in_shadow`, and **59 Tier A pairs** fully in shadow across **7** independent instruments (`1,776.20 deg²·exp`, or **4.27×** POSS-I-E's `415.60 deg²·exp` and **2 single Palomar plates**).
9. **Stratified Blind Pilot Pool:** Built a pool of **16,723** distinct-plate, strictly out-of-shadow (`shadow_fraction == 0` at all three altitudes) Tier A pairs and drew a stratified **200-pair** pilot sample (`data/processed/pilot_sample_pairs.csv`).

---

## Phase 2a: Blind Pilot (Completed — Strictly Out-of-Shadow Plates Only)

10. **Per-Plate Recalibration, QC & Contamination Filters (`src/pilot.py`, `src/calibrate.py`, `src/asteroids.py`):**
    - **Full Per-Plate Archival Saving:** Saved complete `sep.extract` source catalogs (`data/processed/pilot_catalogs/<exposure_id>.parquet`) plus plate footprints, refit WCS, and `8 × 8` calibrated local limiting magnitude grids (`data/processed/pilot_footprints.parquet`).
    - **Astrometric Refit (`src/calibrate.py`):** Propagated Gaia DR3 (`gaia.dr3lite`) proper motions from J2016.0 (`MJD 57388.0`) to `mjd_mid`; ran iterative TAN-SIP polynomial refit (`30"` median pre-shift -> `10" -> 5" -> 3"`, selecting order 3–5 via 5-fold CV).
    - **Photometric Calibration (`src/calibrate.py`):** Fit per-plate monotonic `PchipInterpolator` spline from instrumental mag to Gaia `BP` or `G` (smaller MAD) on matched unsaturated stars; recomputed `8 × 8` local `m_lim` grid from calibrated count turnover (`mode + 0.25 mag`).
    - **Strict Refit QC Gate:** Required bright stars (`mag_cal < m_lim_cal - 2.0`, fallback to top 100 if `< 5` and `n_psf >= 20`) to match Gaia DR3 **`>= 90%` within `3.0"`** AND **`median_resid <= 1.5"`**.
    - **Partner-Detectability Rule:** Candidate must fall inside partner plate's footprint, `>= 2.0 PSF` widths from edges, and `>= 1.0 mag` brighter than partner's **local** calibrated limiting magnitude.
    - **Known-Asteroid Removal (`src/asteroids.py`):** One `astroquery.imcce.Skybot.cone_search` per plate at `mjd_mid` (verified to `0.66"–1.45"` vs JPL Horizons at `1954-05-01` for Ceres, Pallas, Vesta); match radius = `hypot(15", 0.5 * motion_as_hr * exptime_hr) + 5"`.
11. **Phase 2a Blind Pilot Results (`200 / 200` pairs `ok`, `0` failures):**
    - **QC Pass Rate:** `139 / 373` unique plates passed (`234` dropped); **`61 / 200` pairs** (`122` plate-exposures, `5,280.76 deg²`) had both plates pass QC.
    - **Clean Candidate Rate:** `4,073` rule-passing candidates on QC-passed pairs (`5` matched SkyBoT asteroids -> **`4,068` clean candidates** = **`33.3443 /plate`**, **`0.77034 /deg²`**).
    - **Updated Power Analysis (`m_flash >= 9.0`):** Shadow area `2,132.99 deg²·exp`, expected null counts `1,643.1`, MDD = `0.062` (80% power at `alpha=0.05` one-sided), `0.074` (`3-sigma`), `0.124` (`5-sigma`).
    - **Pre-Registration Locked:** Parameters, unblinding QC-fairness rule, measured rates, and commit hash `6ccc16d4e16677947a1fface04a216f8da13bcb1` locked in `reports/preregistration_draft.md` before touching any shadow plates.
12. **Amendment 1 (Pre-Unblinding Design Revision, Cutout Audit & Empirical Power Calibration):**
    - **Across-Plate NB GLM Power Check (`src/power_calibration.py`):** Revealed massive overdispersion (`alpha_hat = 2.6717`, Poisson `chi2/df = 158.10`), yielding `23.6%` power at a `30%` deficit and `8.4%` analytic FPR.
    - **Asteroid Recovery Check:** Caught `2/123` plates with collapsed `PchipInterpolator` photometry (`IQR < 0.5 mag`); added linear-ZP fallback guard in `src/calibrate.py`. Bright asteroid recovery (`V < min(local_mlim, m_lim_archive) - 1.0`, `V > 0`) is `8/9` (`88.9%`).
    - **150-Candidate Cutout Audit (`src/audit_cutouts.py`):** Rendered 60″ boxes across 84 out-of-shadow plates (`reports/figures/pilot_audit_contact_sheet_{1,2,3}.png`). Breakdown: `50.7%` blends, `28.0%` stellar-looking unexplained, `11.3%` plate defects, `6.0%` scanner artifacts, `4.0%` bright-star halos/spikes. Added frozen morphology/proximity/all-source-partner filter (`src/test_artifact_filter.py`), cutting candidates from `4,068` to `1,689` (`chi2/df` `158.10 -> 47.38`).
    - **Revised Primary Design (`src/within_plate_power.py`):** Switched primary test to within-plate conditional Mantel-Haenszel (`0.05 < shadow_fraction < 0.95`) with per-instrument sunlit spatial defect profiles and `>= 1,000`-run placebo rank inference (`FPR ~ 5%` by construction).
    - **Go/No-Go Result:** `Tier A m_flash >= 9.0` (`27.4%` power at `30%` deficit) and `Tier A+B m_flash >= 9.0` (`48.3%`) do not reach `80%`; expanding to **`Tier A m_flash >= 8.0`** (`107.6` expected QC-passing partial-shadow plates) reaches **`89.2%` power** at `30%` deficit (`98.8%` at `40%`).
    - **PDF Exported:** Saved `reports/preregistration_amendment1.pdf` for public OSF/Zenodo timestamping prior to any unblinding.

---

## Phase 2b: Unblinded Earth-Shadow Test (Pending Phase 2a Freeze & User Sign-Off)

- *To be updated after Phase 2a completes and pre-registration is locked.*
