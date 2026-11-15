# Pre-Registration, Amendment 1
## Do short star-like flashes disappear inside Earth's shadow on pre-Sputnik photographic plates?

**Tycho Collins (independent researcher)**  
**Frozen 15 November 2026, 18:15 UTC, before any shadow data were accessed**

---

## 1. Summary

Some studies of old sky survey plates (for example, the VASCO project) report short-lived, star-like flashes on photographic plates taken before the first artificial satellite (Sputnik, 4 October 1957). If any of these flashes were sunlight reflecting off objects in high orbit, they should be missing where Earth's shadow falls at that altitude. This study tests for that deficit.

We compare the number of flash candidates inside and outside Earth's shadow cone at geostationary altitude (42,164 km) on plates taken before 4 October 1957. Each plate is compared with itself: the part of the plate inside the shadow versus the part outside it.

This document fixes the analysis plan, the decision rule, and all filter settings **before** any plate that touches the shadow has been looked at. Amendment 1 records what we learned from sunlit plates only, and the changes made as a result.

---

## 2. Blinding

No exposure with any part of its field inside the shadow (shadow fraction above zero at 42,164 km, 36,000 km or 20,000 km) has been downloaded, opened, queried for cutouts, or processed. We checked that the maximum shadow fraction is exactly 0.0 across all 773 exposures (769 unique plates) used so far, in the 200-pair pilot and the 400-plate sunlit calibration run.

---

## 3. What changed in Amendment 1, and why

All four changes were based on sunlit plates only.

**3.1 Plates differ a lot from each other, so each plate is compared with itself.** The 200-pair pilot showed much larger plate-to-plate variation in candidate counts than a Poisson model allows (negative binomial alpha = 2.6717; Poisson chi-squared per degree of freedom = 158.10). A test that compares plates against each other would have only 23.6% power at a 30% deficit. The primary test therefore conditions on each partially shadowed plate's own candidate count (shadow fraction between 0.05 and 0.95), which removes plate-to-plate and batch differences.

**3.2 About half of single-plate candidates are faint stars, which dilutes any signal.** We processed 400 sunlit plates at flash magnitude 8.0 or deeper (209 of 400, 52.3%, passed quality checks, across 73 batches; 348 unique passing plates when combined with the 139 pilot plates). We checked 3,353 candidates on 54 plate pairs against a second, equally deep or deeper plate of the same sky in the same band. 46.9% (1,573) matched a faint or blended star on the second plate within 15.0 arcseconds; 53.1% (1,780) were truly absent. A faint star cannot vanish in Earth's shadow, so a real deficit in single-plate candidates is diluted by a factor of 0.531 (before depth scaling). The strict paired test (Section 5) removes these stars entirely, but it has fewer plates (about 66 expected, 4.01 events per plate) and can only reliably detect deficits of about 55 to 65%.

**3.3 A human checked that the paired test really removes stars.** In a blind check of 50 cutouts, a person graded whether anything was present on the second plate at the candidate position. Result: 0 of 50 "yes" (leak rate 0%; one-sided 95% upper bound 5.8%; two-sided 95% interval 0% to 7.1%), 4 of 50 "near" (#07 at 15.44", #25 at 19.20", #28 at 21.45" and 29.86", #40 at 17.16", all outside the 15.0" veto radius) and 46 of 50 "no". Labels are in `data/processed/blind_human_audit_50_v2_human_labels.csv`.

Only this partner-plate check was done by a person. The candidate shape and morphology cuts (ellipticity, size, concentration) and the visual defect categories were made by automated software and AI-assisted review, and have **not** been checked by a person.

**3.4 Two safeguards against false signals.**
- *Placebos come only from the real shadow path, shifted in time.* Real partially shadowed plates lie close to the ecliptic (ecliptic latitude within 16.64°, median 4.12°, across the 291 exposures at magnitude 9.0 or deeper). Random sky positions would instead sample star-crowded regions such as the Magellanic Clouds and the Milky Way (ecliptic latitude about 65° to 89°), giving misleading thresholds (critical z = -4.967 at magnitude 8.0). All placebos are therefore made by shifting the real 3D shadow geometry at 42,164 km along the ecliptic in time (at least 15 days from the true date, Sun below -12°, pointing above 10° altitude, shadow fraction 0.05 to 0.95). No random sky positions or random angles are used.
- *Star density is built into the expected in-shadow share.* Because 46.9% of candidates are faint stars, and star density varies across a plate, the expected share of candidates inside the shadow includes local Gaia DR3 star density on a 32 x 32 grid (smoothed at 1.5 cells). This tightens the critical z from -2.163 to -1.956 at magnitude 9.0, and from -1.780 to -1.652 at magnitude 8.0.

---

## 4. Primary test

**Sample.** Every exposure that passes quality checks (Tier A and B), reaches flash magnitude 9.0 or deeper (0.5 s), and is partly in the geostationary shadow (shadow fraction between 0.05 and 0.95 on the unmasked 32 x 32 grid). This is 291 archive exposures, with about 106 expected to pass quality checks.

**Candidates.** Final Frozen Filter v1.2 in single-plate mode (Appendix A).

**Expected share inside the shadow.** Each plate is divided into a 32 x 32 grid. Masked cells (plate border, bright-star halos, close neighbours) are removed. Each remaining cell `c` on plate `i` gets a weight

$$w_{i,c} \propto \left[ 0.469\,G_{i,c} + 0.531\,(P_{\text{inst}(i),c}\,D_{i,c})_{\text{norm}} \right] M_{i,c}, \qquad \sum_c w_{i,c} = 1,$$

where $G$ is local Gaia DR3 star density to the plate's depth, $P$ is the instrument's defect pattern (8 x 8, fitted on sunlit plates only), $D_{i,c} = 10^{0.2\,\text{clip}(m_{\text{lim},i,c} - m_{\text{lim,cal},i},\,-1.5,\,1.5)}$ is local detectability, and $M$ marks unmasked cells. The expected in-shadow share is $p_i = \text{clip}\!\left(\sum_{c \in \text{shadow}} w_{i,c},\, 0.02,\, 0.98\right)$.

**Test statistic.** With $n_i$ candidates on plate $i$, of which $n_{\text{in},i}$ fall in the shadow:

$$Z = \frac{\sum_i (n_{\text{in},i} - n_i p_i)}{\sqrt{\sum_i n_i p_i (1 - p_i)}}, \qquad RR_{MH} = \frac{\sum_i n_{\text{in},i}(1 - p_i)}{\sum_i n_{\text{out},i}\,p_i}.$$

**Significance.** The one-sided p-value is the rank of the observed $Z$ among at least 1,000 placebo values computed on sunlit near-ecliptic plates, matched to the observed shadow-fraction distribution and balanced across the two sides of the shadow chord. Significant means $p \le 0.05$.

**Power (stated before unblinding).** Allowing for the 0.531 star dilution and depth scaling, power to detect a 30% deficit is **77.9%** (94.7% without depth scaling); at a 40% deficit it is 94.3% (99.9%). The placebo false-positive rate is 3.9%. This is 2.1 percentage points below the usual 80% target. We proceed anyway, without adding plates after the fact, and state it here in advance.

---

## 5. Key secondary test: strict paired plates

**Sample.** Every quality-passing Tier A exposure at flash magnitude 8.0 or deeper with a second plate of the same sky taken 5 minutes to 24 hours apart, with identical filter, plate class and calibrated band. The candidate must pass all partner checks: nothing on the second plate within 15.0", at least one source within 120.0", local registration score of at least 0.80 within 60", and the second plate at least 1.0 magnitude deeper at that spot. This is 253 archive exposures, about 66 expected to pass.

**Role.** Removing all stars leaves few plates (4.01 events per plate), so power at a 30% deficit is **only 19.4%** (22.3% without depth scaling); 69.7% at 60% and 86.0% at 70%. The smallest deficit it can detect with 80% power is about 55 to 65% (unscaled). This test is used only to check that the effect points the same way (rate ratio below 1). It is not a standalone test.

---

## 6. Decision rule

A shadow deficit will be reported as evidence for a pre-Sputnik population of orbital flashes **only if all three of the following hold**:

**(a) The primary test is significant:** $Z \le \text{critical } z$ (placebo $p \le 0.05$, one-sided) against at least 1,000 time-shifted ecliptic shadow placebos.

**(b) All three negative controls pass:**
1. *Shifted dates:* repeating the test with dates shifted by $\pm 1$ to $\pm 5$ days and by $\pm 30$, $\pm 60$ and $\pm 90$ days gives the strongest deficit near zero shift and no significant deficit ($p > 0.05$) away from it.
2. *Fake altitudes:* comparing the shadow cone at 42,164 km with 36,000, 20,000 and 5,000 km behaves as expected for objects at high altitude, not as a fixed pattern on the plate.
3. *Same local time, no shadow:* sunlit plates taken near local midnight (within 2 hours), with no shadow and ecliptic latitude between 5° and 18°, just outside the $\pm 4.35°$ shadow cone, show no deficit.

**(c) The strict paired test points the same way:** rate ratio below 1 and $Z$ below 0 (it does not need to be significant on its own).

If (a) fails, the result is **null**. If (a) passes but (b) or (c) fails, the result is **inconclusive or caused by an artefact**.

---

## 7. Other planned analyses

- **Multi-epoch star veto (sensitivity analysis, not required before unblinding).** For surviving candidates on partially shadowed plates, check deep plates of the same position from any date (StarGlass or APPLAUSE cutouts) and remove any candidate that appears there. This removes faint stars directly instead of modelling the 46.9% share.
- **Deeper single-plate variant.** The same test at flash magnitude 8.0 or deeper (817 archive exposures, about 314 expected to pass).
- **Dose-response.** Does the deficit grow with shadow fraction, plate depth and shadow altitude (42,164, 36,000 and 20,000 km)?
- **Across-plate model.** A negative binomial mixed model of clean candidate rate on sunlit versus shadowed plates, with exposed area as offset, instrument as a fixed effect, ecliptic latitude, solar elongation, hours from local midnight, airmass, and a random effect for batch (batches defined by plate serial within 50 and date within 30 days, within archive and series; sunlit batch ICC = 20.9% across 73 batches).
- **Aligned flashes (exploratory).** A search for 3 or more flashes in a straight line (within 1.0 FWHM) and roughly evenly spaced (successive gaps within a ratio of 0.75 to 1.25), compared with at least 1,000 shuffles per plate (events de-clustered at 30"). The geostationary drift direction test uses only plates where that direction is at least 20° from the scan axes, with APPLAUSE plates scanned in both directions. On sunlit plates this search found nothing unusual: 4 observed versus 5.36 expected on the 102 pilot plates ($p = 0.774$); 71 versus 80.21 expected on 209 plates (95th percentile 97.0, $p = 0.843$); 0 versus 0.13 expected on the 12 plates with a clear drift direction.

---

## 8. Calibration and power (sunlit plates only)

From 1,000 null and 1,000 signal simulations using time-shifted shadow geometry on sunlit plates (32 near-ecliptic plates at magnitude 9.0; 61 at magnitude 8.0; 102 paired pilot plates). Power is shown as depth-scaled / unscaled; FPR is the false-positive rate. Full results: `data/processed/final_frozen_power_calibration.csv`.

| Test | Plates | Gaia | Crit. $z$ | FPR (analytic / placebo) | Power at 30% | Power at 40% |
|---|---:|:---:|---:|---:|---:|---:|
| **Primary: single plate, mag $\ge 9.0$** | 106 | Yes | $-1.956$ | 6.7% / 3.9% | 77.9% / 94.7% | 94.3% / 99.9% |
| Primary, no Gaia density | 106 | No | $-2.163$ | 7.2% / 3.4% | 74.6% / 93.9% | 93.2% / 99.9% |
| **Key secondary: strict paired, mag $\ge 8.0$** | 66 | No | $-2.035$ | 8.0% / 3.8% | 19.4% / 22.3% | 24.2% / 38.3% |
| Secondary: single plate, mag $\ge 8.0$ | 314 | Yes | $-1.652$ | 5.7% / 5.6% | 95.2% / 99.5% | 99.6% / 100.0% |
| Secondary, no Gaia density | 314 | No | $-1.780$ | 7.1% / 6.0% | 94.0% / 99.5% | 99.5% / 100.0% |

The analytic false-positive rate uses the standard threshold of $-1.645$. Without star dilution, the primary test has 100% power at a 30% deficit. Adding Gaia density tightens the primary threshold by 0.21. For comparison, random-sky placebos would have given a threshold of $-4.967$ at magnitude 8.0.

---

## 9. Reporting after unblinding

Shadowed and sunlit plates are processed by the same code. Alongside the plate count, we will report the quality-check pass rates for shadowed and sunlit plates, and the number of distinct instruments, plate series and batches in the unblinded sample, including the share of the largest and second-largest batch.

---

## Appendix A. Frozen filter settings (Final Frozen Filter v1.2)

- **Border mask:** the larger of 5% of width or height and 4.0 x median PSF radius, plus zero-valued pixels, removed on the 32 x 32 grid.
- **Polarity and pixel checks:** image inverted automatically if `(p50 - p05) > 1.3 x (p99.5 - p50)` on the valid-pixel percentiles; a candidate needs peak S/N $\ge 4.0$, minimum-of-3 S/N $\ge -3.0$, and no saturated pixels in the central 5 x 5.
- **Background and extraction:** background with box size 128 x 128 and filter 3 x 3 (`sep.Background`); extraction at 6.0 x rms, minimum area 7 pixels, no deblending (`sep.extract`, `deblend_nthresh = 1`, `deblend_cont = 1.0`).
- **Shape and proximity cuts (automated, not checked by a person):** `flag = 0`, `npix >= 5`, `flux > 0`, no Gaia source within 3.0", ellipticity $\le 0.20$, PSF radius between 0.7 and 1.5 x the plate median, 9 x 9 peak concentration $\ge 0.03$, bright-star halo mask (stars brighter than the calibrated limit minus 3.0 mag, radius 20" to 60"), 2.0" to 15.0" neighbour veto (area removed on the 32 x 32 grid), SkyBoT asteroid veto, and 30.0" de-clustering.
- **Astrometry and photometry (`src/calibrate.py`):** Gaia DR3 positions propagated to the mid-exposure date; iterative TAN-SIP fit (30", 10", 5", 3"; order 3 to 5 chosen by 5-fold cross-validation); monotonic PCHIP calibration to Gaia BP or G (with a linear zero-point fallback guard, IQR $\ge 0.5$ mag); 8 x 8 grid of local limiting magnitude. A plate passes quality checks if at least 90% of bright stars match Gaia within 3.0" and the median residual is at most 1.5".
- **Paired partner rules (human-checked, 0 of 50 leaks):** same archive filter, plate class and calibrated band; partner at least 4.0 PSF widths from its edge; partner local limit at least 1.0 mag deeper than the candidate; no partner source within 15.0"; at least one partner source within 120.0"; 60" registration score at least 0.80 (match within 5.0" when there are at least 3 neighbours).

---

## Appendix B. Frozen files and commits

- **Repository:** https://github.com/tychocollins/presputnik-technosignatures
- **Frozen analysis code and calibration commit:** `a41c00421f968b1c0c46f75c7c765b4406da72c0` (builds on Phase 2a commit `6ccc16d4e16677947a1fface04a216f8da13bcb1`)
- **Frozen files:**
  - `src/final_frozen_calibration.py`
  - `src/aligned_tracks.py`
  - `src/calibrate.py`
  - `src/declump_and_mask.py`
  - `src/process_sunlit_m8.py`
  - `data/processed/blind_human_audit_50_v2_human_labels.csv`
  - `data/processed/final_frozen_power_calibration.csv`
- **Human audit figure:** `reports/figures/blind_human_audit_50_v2.png`
