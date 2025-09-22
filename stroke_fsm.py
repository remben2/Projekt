from __future__ import annotations

from collections import deque
from typing import Optional, Tuple
from constants import TRUNK_DERIV_THR, SPM_MIN_S, SPM_MAX_S


class StrokeFSM:
    """
    Csapásállapot és metrikák:
      - drive_ms, recovery_ms, ratio
      - spm (finish→finish és catch→catch)
      - trunk_max (Drive alatt, csúcs a végén)
      - catch-depth: knee_min, trunk_forward_max
      - flags: Ratio/SPM/Trunk
      - score: 0..100 (ratio 40, spm 30, posture 30)
    A logot a következő Catch belépésekor írjuk (ekkor teljes a recovery).
    """
    def __init__(self):
        self.state = "Recovery"
        self.t0: Optional[float] = None
        self.t_drive: Optional[float] = None
        self.last_finish: Optional[float] = None

        self.spm_window = deque(maxlen=7)
        self.drive_ms = 0.0
        self.recovery_ms = 0.0
        self.ratio = 0.0
        self.spm = 0.0

        self.trunk_max = 0.0
        self.last_trunk_max = 0.0
        self.neg_frames = 0

        self.pending_drive_ms: Optional[float] = None

        # catch-depth proxyk
        self.knee_min: Optional[float] = None
        self.trunk_forward_max = 0.0
        self.last_catch_knee: Optional[float] = None
        self.last_catch_trunk: Optional[float] = None

        self.last_flags: Tuple[str, str, str] = ("", "", "")
        self.last_score: Optional[Tuple[float, float, float, float]] = None
        self.last_catch_time: Optional[float] = None

        # Back metrics (utolsó teljes stroke-hoz tartozó értékek)
        self.last_back_curve: Optional[float] = None
        self.last_back_bend: Optional[float] = None

        self.log = []

    # ----- minőségcímkék -----
    def quality_flags(self, ratio=None, spm=None, trunk_max=None) -> Tuple[str, str, str]:
        r = float(self.ratio if ratio is None else ratio)
        s = float(self.spm if spm is None else spm)
        t = float(self.last_trunk_max if trunk_max is None else trunk_max)
        rf = "Ratio OK" if 0.4 <= r <= 0.7 else ("Ratio low" if r < 0.35 else ("Ratio high" if r > 0.80 else "Ratio borderline"))
        sf = "SPM OK"   if 18  <= s <= 32  else ("SPM low"   if s < 16     else ("SPM high"  if s > 34    else "SPM borderline"))
        tf = "Trunk OK" if 20  <= t <= 35  else ("Trunk low" if t < 15     else ("Trunk high" if t > 40   else "Trunk borderline"))
        return rf, sf, tf

    # ----- trunk követés -----
    def enter_drive(self):
        self.trunk_max = 0.0

    def update_trunk(self, trunk_deg: float, trunk_der: float, der_thr: float = TRUNK_DERIV_THR):
        if self.state == "Drive" and float(trunk_der) > der_thr:
            self.trunk_max = max(self.trunk_max, float(trunk_deg))

    # ----- catch-depth gyűjtés -----
    def update_catch_proxies(self, knee_deg: float, trunk_deg: float):
        if self.state == "Catch":
            kd = float(knee_deg); td = float(trunk_deg)
            self.knee_min = kd if self.knee_min is None else min(self.knee_min, kd)
            self.trunk_forward_max = max(self.trunk_forward_max, td)

    # ----- stroke score -----
    def stroke_score(self):
        # Ratio (0–40)
        r = float(self.ratio)
        if 0.4 <= r <= 0.7:
            ratio_pts = 40.0
        else:
            d = min(abs(r - 0.55), 0.35)
            ratio_pts = max(0.0, 40.0 * (1.0 - d / 0.35))
        # SPM (0–30)
        s = float(self.spm)
        if 18.0 <= s <= 32.0:
            center, span = 25.0, 7.0
            spm_pts = max(0.0, 30.0 * (1.0 - (abs(s - center) / span) ** 2))
            spm_pts = max(spm_pts, 24.0)
        else:
            d = min(abs(s - 25.0), 13.0)
            spm_pts = max(0.0, 24.0 * (1.0 - d / 13.0))
        # Testhelyzet (0–30)
        tmax = float(self.last_trunk_max if self.last_trunk_max is not None else 0.0)
        cknee = float(self.last_catch_knee if self.last_catch_knee is not None else 65.0)
        ctrunk = float(self.last_catch_trunk if self.last_catch_trunk is not None else 20.0)
        def band_score(x, lo, hi, span):
            if lo <= x <= hi: return 1.0
            if x < lo:        return max(0.0, 1.0 - (lo - x) / span)
            return max(0.0, 1.0 - (x - hi) / span)
        trunk_ok = band_score(tmax, 20, 35, span=15)
        knee_ok  = band_score(cknee, 50, 70, span=20)
        ctrk_ok  = band_score(ctrunk, 15, 30, span=15)
        posture_pts = 30.0 * (0.5 * trunk_ok + 0.35 * knee_ok + 0.15 * ctrk_ok)
        s_pts = spm_pts
        p_pts = posture_pts
        total = round(ratio_pts + s_pts + p_pts, 1)
        return total, ratio_pts, s_pts, p_pts

    # ----- állapotgép -----
    def update(self, knee_ddeg: float, drive_on: bool, finish_on: bool, knee_deg: float, t: float):
        now = float(t)
        changed = False

        # Recovery -> Catch
        if self.state == "Recovery":
            if abs(knee_ddeg) < 5.0 and knee_deg <= 65.0:
                # előző stroke lezárása (itt már ismert a recovery)
                if self.last_finish is not None and self.pending_drive_ms is not None:
                    self.recovery_ms = max((now - self.last_finish) * 1000.0, 1.0)
                    self.drive_ms = max(self.pending_drive_ms, 1.0)
                    self.ratio = self.drive_ms / self.recovery_ms if self.recovery_ms > 1e-3 else 0.0

                    flags = self.quality_flags(self.ratio, self.spm, self.last_trunk_max)
                    self.last_flags = flags

                    score, r_pts, s_pts, p_pts = self.stroke_score()
                    self.last_score = (score, r_pts, s_pts, p_pts)

                    ck = float(self.last_catch_knee) if self.last_catch_knee is not None else float('nan')
                    ct = float(self.last_catch_trunk) if self.last_catch_trunk is not None else float('nan')

                    self.log.append([
                        self.drive_ms, self.recovery_ms, self.ratio, self.spm,
                        self.last_trunk_max, flags[0], flags[1], flags[2],
                        ck, ct, score, r_pts, s_pts, p_pts,
                        (float('nan') if self.last_back_curve is None else float(self.last_back_curve)),
                        (float('nan') if self.last_back_bend is None else float(self.last_back_bend)),
                    ])
                    self.pending_drive_ms = None

                # Catch→Catch SPM
                if self.last_catch_time is not None:
                    T = now - self.last_catch_time
                    if SPM_MIN_S <= T <= SPM_MAX_S:
                        self.spm_window.append(T)
                        mean_T = sum(self.spm_window) / len(self.spm_window)
                        self.spm = 60.0 / mean_T if mean_T > 1e-6 else 0.0
                self.last_catch_time = now

                # következő stroke-hoz nullázás
                self.knee_min = None
                self.trunk_forward_max = 0.0

                self.state = "Catch"
                self.t0 = now
                changed = True

        # Catch -> Drive
        if self.state == "Catch" and drive_on:
            self.state = "Drive"
            self.t_drive = now
            self.enter_drive()
            # fogás proxyk rögzítése ehhez a stroke-hoz
            self.last_catch_knee = self.knee_min
            self.last_catch_trunk = self.trunk_forward_max
            changed = True

        # Drive -> Finish
        if self.state == "Drive" and (finish_on or knee_ddeg < 15.0):
            self.state = "Finish"
            changed = True

        # Finish -> Recovery (itt nincs log, csak rögzítések)
        if self.state == "Finish":
            self.neg_frames = self.neg_frames + 1 if knee_ddeg < -15.0 else 0
            if self.neg_frames >= 2:
                if self.t_drive is not None:
                    drive = (now - self.t_drive) * 1000.0
                    self.pending_drive_ms = max(drive, 1.0)

                if self.last_finish is not None:
                    dur = now - self.last_finish
                    if SPM_MIN_S <= dur <= SPM_MAX_S:
                        self.spm_window.append(dur)
                if len(self.spm_window) >= 1:
                    mean_T = sum(self.spm_window) / len(self.spm_window)
                    self.spm = 60.0 / mean_T if mean_T > 1e-6 else 0.0

                self.last_finish = now
                self.last_trunk_max = self.trunk_max
                # a Finish→Recovery átmenetnél nem változtatunk a back metrikákon;
                # azokat a fő ciklus frissíti folyamatosan

                self.state = "Recovery"
                self.t0 = None
                self.t_drive = None
                self.neg_frames = 0
                changed = True

        return changed
