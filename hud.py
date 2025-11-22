from __future__ import annotations

from typing import Tuple
import cv2

# HUD pozíciókonstansok
HUD_X = 30
HUD_Y0 = 60
HUD_DY = 35

def draw_hud(image, fsm, angles: Tuple[float, float, float, float], knee_ddeg: float) -> None:
    """
    Egységes HUD kirajzolás a bal felső és bal alsó blokkokkal.

    angles: (angle_elbow, angle_knee, angle_hip, angle_trunk)
    knee_ddeg: térd szögsebesség deg/s
    """
    angle_elbow, angle_knee, angle_hip, angle_trunk = angles
    x, y = HUD_X, HUD_Y0

    def put(txt, col=(255, 255, 255)):
        nonlocal y
        y += HUD_DY
        cv2.putText(image, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

    # Bal felső – ízületi szögek + térd derivált
    y = 25
    put(f"Left elbow: {int(round(angle_elbow))} deg")
    put(f"Left knee:  {int(round(angle_knee))} deg")
    put(f"Left hip:   {int(round(angle_hip))} deg")
    put(f"Knee dθ/dt: {int(round(knee_ddeg))} deg/s", (255, 220, 180))

    # Alsó blokk – állapot, idők, ratio/spm
    y = 365
    put(f"State: {fsm.state}", (255, 255, 180))
    put(f"Drive: {int(fsm.drive_ms)} ms  Rec: {int(fsm.recovery_ms)} ms", (255, 255, 180))
    put(f"Ratio D:R = {fsm.ratio:.2f}   SPM ~ {fsm.spm:.1f}", (255, 255, 180))
    trunk_hud = fsm.last_trunk_max
    put(f"Trunk max: {int(trunk_hud)} deg", (200, 255, 255))

    # Minőségcímkék
    rf, sf, tf = fsm.last_flags
    put(f"{rf} | {sf} | {tf}", (180, 255, 180))

    # Catch-depth
    ck = 0 if fsm.last_catch_knee is None else int(fsm.last_catch_knee)
    ct = 0 if fsm.last_catch_trunk is None else int(fsm.last_catch_trunk)
    put(f"Catch knee min: {ck} deg", (180, 220, 255))
    put(f"Catch trunk: {ct} deg", (180, 220, 255))

    # Score
    if fsm.last_score is not None:
        # Ha hátgörbület score is van, jelenítsük meg
        if len(fsm.last_score) == 4:
            sc, rpts, spts, ppts = fsm.last_score
            put(f"Score: {sc:.1f} (R {rpts:.0f} | S {spts:.0f} | P {ppts:.0f})", (255, 220, 120))
        elif len(fsm.last_score) == 5:
            sc, rpts, spts, ppts, back_curve_pts = fsm.last_score
            put(f"Score: {sc:.1f} (R {rpts:.0f} | S {spts:.0f} | P {ppts:.0f} | B {back_curve_pts:.0f})", (255, 220, 120))
