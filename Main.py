"""
Ergo stroke-analízis (MediaPipe + OpenCV)
- Tiszta StrokeFSM: ratio, SPM, trunk_max, catch-depth, flags, score
- Stabil időalap: videó idő (CAP_PROP_POS_MSEC), fallback fps
- Jól tagolt feldolgozó ciklus és egységes HUD
Futtatás:
  python3 teszt/Teszteles.py --video "/path/to/video.mp4" --side bal
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from hud import draw_hud
from utils_geom import (
    calculate_angle,
    trunk_angle_deg,
    get_xy,
    back_contour_from_mask,
    back_curvature_metric,
    back_bend_angle,
    back_contour_from_edges,
)
from utils_filters import EMASmoother, WindowDerivative, WindowPeak
from stroke_fsm import StrokeFSM
from constants import (
    ENTER_THR, EXIT_THR, HOLD_N,
    SPM_MIN_S, SPM_MAX_S,
    TRUNK_DERIV_THR,
    ALPHA_ELBOW, ALPHA_KNEE, ALPHA_HIP, ALPHA_TRUNK, ALPHA_BACK,
    CURVE_WARN, CURVE_ALERT, BEND_WARN, BEND_ALERT,
    HANDLE_SIGMA_WARN, HANDLE_SIGMA_ALERT, HANDLE_RMS_WARN, HANDLE_RMS_ALERT,
    HANDLE_SIGMA_ALPHA, HANDLE_RMS_ALPHA,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose




@dataclass
class DriveCue:
    on: bool = False
    above: int = 0
    below: int = 0

def update_drive_cue(cue: DriveCue, knee_ddeg: float) -> None:
    if knee_ddeg > ENTER_THR:
        cue.above += 1; cue.below = 0
    elif knee_ddeg < EXIT_THR:
        cue.below += 1; cue.above = 0
    if not cue.on and cue.above >= HOLD_N:
        cue.on = True
    if cue.on and cue.below >= HOLD_N:
        cue.on = False

def export_csv(fsm: StrokeFSM, path: str = "strokes.csv") -> None:
    if not fsm.log:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "drive_ms","recovery_ms","ratio","spm","trunk_max_deg",
            "ratio_flag","spm_flag","trunk_flag",
            "catch_knee_min_deg","catch_trunk_deg",
            "score_total","score_ratio","score_spm","score_posture","score_back_curve"
        ])
        w.writerows(fsm.log)
    print(f"Mentve: {path}   sorok: {len(fsm.log)}")

def video_feldolgozas(video_path: str, side: str = "bal"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót.")
        return

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True,
        smooth_segmentation=True,
    ) as pose:

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / float(fps)
        # Nyél nyomvonal hossza ~0.8s (korlátok között)
        trail_len = max(10, min(300, int(0.8 / dt)))

        elbow_s = EMASmoother(ALPHA_ELBOW)
        knee_s  = EMASmoother(ALPHA_KNEE)
        hip_s   = EMASmoother(ALPHA_HIP)
        trunk_s = EMASmoother(ALPHA_TRUNK)
        knee_der = WindowDerivative(window=3, dt=dt)
        back_s  = EMASmoother(ALPHA_BACK)
        bend_s  = EMASmoother(0.45)  # enyhe simítás a derék szögre
        # EMA-k a nyél metrikákhoz
        sigma_s    = EMASmoother(HANDLE_SIGMA_ALPHA)
        rms_drv_s  = EMASmoother(HANDLE_RMS_ALPHA)
        rms_rcv_s  = EMASmoother(HANDLE_RMS_ALPHA)

        trunk_peak   = WindowPeak(window=7, mode="max", tol=4.0)
        elbow_trough = WindowPeak(window=7, mode="min", tol=6.0)

        finish_on = False
        finish_flash = 0
        last_drive_on = False
        finish_ready = False
        FINISH_FLASH_FRAMES = 8

        prev_trunk: Optional[float] = None
        cue = DriveCue()
        fsm = StrokeFSM()
        # Korábbi egyszerű verzióhoz visszaállítva: nincs kontúr-tartás és hiszterézis
        # Nyél-nyomvonal tároló (pixel koordináták)
        handle_trail = deque(maxlen=trail_len)
        handle_trail_rel_y = deque(maxlen=trail_len)  # normalizált y (csípőhöz viszonyítva)
        handle_trail_rel_xy = deque(maxlen=trail_len) # normalizált (x,y) + fázis
    # Catch dwell: teljesen eltávolítva

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Videó-idő (VFR barát)
            t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            image.flags.writeable = True

            # Első apró lépés a hátgörbülethez: test-szegmentáció vizualizálása
            if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
                try:
                    mask = results.segmentation_mask
                    if mask is not None and mask.shape[:2] == image.shape[:2]:
                        mask_bin = mask > 0.5  # küszöb (később finomítható)
                        tint = np.array([200, 230, 255], dtype=np.uint8)  # világos kékes
                        # Csak a maszk területén halvány színezés
                        image[mask_bin] = (0.6 * image[mask_bin] + 0.4 * tint).astype(np.uint8)
                except Exception:
                    # Régebbi mp verzióknál előfordulhat eltérés – csendben kihagyjuk
                    pass

            if results.pose_landmarks is None:
                cv2.putText(image, "No pose detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("Mediapipe Feed", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            lm = results.pose_landmarks.landmark

            if side.lower() == "bal":
                L_shoulder = get_xy(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
                L_elbow    = get_xy(lm, mp_pose.PoseLandmark.LEFT_ELBOW)
                L_wrist    = get_xy(lm, mp_pose.PoseLandmark.LEFT_WRIST)
                L_hip      = get_xy(lm, mp_pose.PoseLandmark.LEFT_HIP)
                L_knee     = get_xy(lm, mp_pose.PoseLandmark.LEFT_KNEE)
                L_ankle    = get_xy(lm, mp_pose.PoseLandmark.LEFT_ANKLE)
                L_nose     = get_xy(lm, mp_pose.PoseLandmark.NOSE, min_vis=0.2)

                req = [L_shoulder, L_elbow, L_wrist, L_hip, L_knee, L_ankle]
                if any(v is None for v in req):
                    cv2.putText(image, "Landmarks occluded", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                else:
                    # Szögek és simítás
                    angle_elbow = calculate_angle(L_shoulder, L_elbow, L_wrist)
                    angle_knee  = calculate_angle(L_hip, L_knee, L_ankle)
                    angle_hip   = calculate_angle(L_shoulder, L_hip, L_knee)
                    angle_trunk = trunk_angle_deg(L_shoulder, L_hip)

                    elbow_sm = elbow_s.update(angle_elbow)
                    knee_sm  = knee_s.update(angle_knee)
                    hip_sm   = hip_s.update(angle_hip)
                    trunk_sm = trunk_s.update(angle_trunk)

                    # Csúcs/teknő detektálók
                    trunk_is_peak = trunk_peak.update(trunk_sm)
                    elbow_is_trough = elbow_trough.update(elbow_sm)

                    # Trunk derivált
                    trunk_der = 0.0 if prev_trunk is None else (trunk_sm - prev_trunk) / dt
                    prev_trunk = trunk_sm

                    # Finish cue: csak Drive OFF pillanatában vizsgálunk
                    if not last_drive_on and cue.on:
                        finish_ready = True
                        finish_on = False
                        finish_flash = 0
                        trunk_peak.buf.clear()
                        elbow_trough.buf.clear()

                    if last_drive_on and not cue.on and finish_ready:
                        elbow_small = (elbow_sm <= 65.0)
                        turnover = (trunk_der <= 0.0)
                        cond_finish = (trunk_is_peak or turnover) and (elbow_is_trough or elbow_small)
                        if cond_finish:
                            finish_flash = FINISH_FLASH_FRAMES
                        finish_ready = False
                    last_drive_on = cue.on

                    finish_on = finish_flash > 0
                    if finish_flash > 0:
                        finish_flash -= 1

                    # Catch-depth proxyk
                    fsm.update_catch_proxies(knee_deg=knee_sm, trunk_deg=trunk_sm)

                    # Knee derivált és drive cue
                    knee_ddeg = knee_der.update(knee_sm)
                    update_drive_cue(cue, knee_ddeg)

                    # Trunk max frissítés (hátradőlési fázis)
                    fsm.update_trunk(trunk_deg=trunk_sm, trunk_der=trunk_der)

                    # FSM update
                    fsm.update(knee_ddeg=knee_ddeg,
                               drive_on=cue.on,
                               finish_on=finish_on,
                               knee_deg=angle_knee,
                               t=t_video)

                    # Landmarks rajz
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )

                    # HUD
                    draw_hud(image, fsm, (angle_elbow, angle_knee, angle_hip, angle_trunk), knee_ddeg)

                    # Hátgörbe előnézet (piros görbe): maszkból mintavétel a váll–csípő között
                    if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
                        try:
                            # Hát oldal választása: orr (NOSE) melyik oldalán van a váll–csípő vonalnak?
                            # Az orr a test "elülső" oldalán van; a hát a másik oldalon.
                            prefer_sign = None
                            try:
                                sx, sy = L_shoulder
                                hx, hy = L_hip
                                vx, vy = (hx - sx), (hy - sy)
                                nlen = (vx**2 + vy**2) ** 0.5 + 1e-9
                                nx, ny = (-vy / nlen, vx / nlen)
                                if L_nose is not None:
                                    nx0, ny0 = L_nose
                                    # váll pontra viszonyítva a NOSE vektor
                                    dx, dy = (nx0 - sx), (ny0 - sy)
                                    nose_side = dx * nx + dy * ny
                                    # front oldal = sign(nose_side); back = ellenkező előjel
                                    prefer_sign = -1 if nose_side > 0 else +1
                                else:
                                    # Fallback: könyök->csukló vektor alapján
                                    ex, ey = L_elbow
                                    wx, wy = L_wrist
                                    awx, awy = (wx - ex), (wy - ey)
                                    dot = awx * nx + awy * ny
                                    prefer_sign = -1 if dot > 0 else +1
                            except Exception:
                                prefer_sign = None

                            # Kontúr: először a sziluett (edge-based), utána fallback a sugaras maszk-módszer
                            contour = back_contour_from_edges(
                                results.segmentation_mask, L_shoulder, L_hip,
                                prefer_sign=prefer_sign, n_keep=60, thresh=0.45,
                            )
                            if not contour:
                                contour = back_contour_from_mask(
                                    results.segmentation_mask, L_shoulder, L_hip,
                                    n_samples=40, thresh=0.5, max_radius=30, prefer_sign=prefer_sign,
                                )
                            for j in range(1, len(contour)):
                                x1, y1 = contour[j-1]
                                x2, y2 = contour[j]
                                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                            # Gyors görbületi mérőszám
                            if contour:
                                ratio, max_dev = back_curvature_metric(
                                    contour, L_shoulder, L_hip, image.shape
                                )
                                ratio_sm = back_s.update(ratio)
                            else:
                                ratio = 0.0; max_dev = 0.0; ratio_sm = back_s.update(0.0)
                            # Hátgörbület score-hoz mentés
                            fsm.last_back_curve = ratio_sm
                            # Színezés: zöld < narancs < piros (küszöbök constants.py-ből)
                            def pick_color_curve(val: float):
                                if val >= CURVE_ALERT: return (0, 0, 255)  # piros
                                if val >= CURVE_WARN:  return (0, 165, 255)  # narancs
                                return (0, 200, 0)  # zöld

                            # HUD: Back curve
                            color_curve = pick_color_curve(ratio_sm)
                            base_y = 190  # lejjebb vittük, hogy ne takarja a felső szövegeket
                            # Színes pont a text előtt (Back curve)
                            cv2.circle(image, (20, base_y-8), 6, color_curve, -1)
                            cv2.putText(
                                image,
                                f"Back curve: {ratio_sm:.2f} (raw {ratio:.2f}, {int(max_dev)}px)",
                                (35, base_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color_curve,
                                2,
                                cv2.LINE_AA,
                            )

                            # Back bend (szög) + simítás
                            bend_deg_raw = back_bend_angle(contour, L_shoulder, L_hip, image.shape) if contour else 0.0
                            bend_deg = bend_s.update(bend_deg_raw)
                            def pick_color_bend(val: float):
                                if val <= BEND_ALERT: return (0, 0, 255)
                                if val <= BEND_WARN:  return (0, 165, 255)
                                return (0, 200, 0)

                            color_bend = pick_color_bend(bend_deg)
                            # Második sor dot + text
                            cv2.circle(image, (20, base_y+22-8), 6, color_bend, -1)
                            cv2.putText(
                                image,
                                f"Back bend: {int(round(bend_deg))} deg",
                                (35, base_y+22),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color_bend,
                                2,
                                cv2.LINE_AA,
                            )
                            # Opcionális: CSV loghoz aktuális frame back metrikák (egyszerű megjelenítéshez fsm-ben is tárolhatnánk)
                            # (Most csak a konzolba írnék debug célra, ha szükséges később bővíthetjük.)
                            # print(f"DBG_BACK,{t_video:.2f},{ratio_sm:.3f},{bend_deg:.1f}")
                        except Exception:
                            pass

                    # --- NYÉL FIGYELÉS – 1. apró lépés -----------------------------------------
                    # A nyél helyének ideiglenes reprezentációja: bal csukló (L_wrist)
                    # Jelenítsük meg a nyél pozícióját a CSÍPŐHÖZ képest normalizálva
                    # (skála: váll–csípő távolság), így testarányfüggetlen lesz az érték.
                    if L_wrist is not None and L_hip is not None and L_shoulder is not None:
                        H_img, W_img = image.shape[:2]
                        # Pixel koordináták
                        wx = int(round(L_wrist[0] * W_img)); wy = int(round(L_wrist[1] * H_img))
                        hx = int(round(L_hip[0] * W_img));   hy = int(round(L_hip[1] * H_img))
                        sx = int(round(L_shoulder[0] * W_img)); sy = int(round(L_shoulder[1] * H_img))
                        # Skála: váll–csípő távolság (min. 10 px a stabilitásért)
                        scale = max(10.0, float(((sx - hx)**2 + (sy - hy)**2) ** 0.5))
                        rel_x = (wx - hx) / scale
                        rel_y = (wy - hy) / scale

                        # Rajz: csípőnél egy kis kereszt, a "nyélnél" (csukló) sárga pont
                        cv2.line(image, (hx - 6, hy), (hx + 6, hy), (0, 255, 255), 2)
                        cv2.line(image, (hx, hy - 6), (hx, hy + 6), (0, 255, 255), 2)
                        cv2.circle(image, (wx, wy), 6, (0, 255, 255), -1)

                        # Catch zóna és dwell: eltávolítva

                        # Nyomvonal frissítés és kirajzolás
                        handle_trail.append((wx, wy))
                        if len(handle_trail) >= 2:
                            for i in range(1, len(handle_trail)):
                                x1, y1 = handle_trail[i-1]
                                x2, y2 = handle_trail[i]
                                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                        # HUD szöveg a back metrikák alatt
                        hud_y = 190 + 22 * 2 + 8  # back curve + back bend alatt
                        cv2.putText(
                            image,
                            f"Handle rel(hip): x={rel_x:+.2f}, y={rel_y:+.2f}",
                            (35, hud_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        # Waviness (σy) – opció 1: a normalizált y értékek szórása az utóbbi ~0.8s-ban
                        handle_trail_rel_y.append(rel_y)
                        if len(handle_trail_rel_y) >= 3:
                            arr = np.array(handle_trail_rel_y, dtype=np.float32)
                            sigma_y = float(np.std(arr))
                        else:
                            sigma_y = 0.0
                        # EMA kijelzéshez
                        sigma_disp = sigma_s.update(sigma_y)
                        # Szín kiválasztás sigma_y-hoz
                        def pick_color_sigma(val: float):
                            if val >= HANDLE_SIGMA_ALERT: return (0, 0, 255)
                            if val >= HANDLE_SIGMA_WARN:  return (0, 165, 255)
                            return (0, 200, 0)
                        color_sigma = pick_color_sigma(sigma_disp)
                        # Színes pont + felirat
                        cv2.circle(image, (20, hud_y + 22 - 8), 6, color_sigma, -1)
                        cv2.putText(
                            image,
                                f"Handle waviness sigma_y: {sigma_disp:.3f}",
                            (35, hud_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_sigma,
                            2,
                            cv2.LINE_AA,
                        )

                        # Opció 2: Egyenes illesztés a trail-re (norm. rel koordinátákban) és RMS távolság
                        # Tároljuk a (rel_x, rel_y, is_drive) pontokat
                        is_drive = (fsm.state == "Drive")
                        handle_trail_rel_xy.append((float(rel_x), float(rel_y), bool(is_drive)))

                        def rms_line_fit(points: np.ndarray) -> float:
                            # points: (N,2) normalizált koordináták
                            if points.shape[0] < 3:
                                return 0.0
                            mean = points.mean(axis=0)
                            centered = points - mean
                            # PCA: legnagyobb sajátérték sajátvektora a főirány
                            cov = centered.T @ centered / max(1, (points.shape[0] - 1))
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            v = eigvecs[:, np.argmax(eigvals)]  # (2,)
                            # ortogonális eltérés komponense
                            proj = centered @ v
                            recon = np.outer(proj, v)
                            orth = centered - recon
                            rms = float(np.sqrt(np.mean(np.sum(orth*orth, axis=1))))
                            return rms

                        # Szétválogatás Drive / Recovery
                        if len(handle_trail_rel_xy) >= 5:
                            pts = np.array([(x, y) for (x, y, _) in handle_trail_rel_xy], dtype=np.float32)
                            flags = np.array([int(d) for (_, _, d) in handle_trail_rel_xy], dtype=np.int32)
                            drv_pts = pts[flags == 1]
                            rcv_pts = pts[flags == 0]
                            rms_drv = rms_line_fit(drv_pts) if drv_pts.size else 0.0
                            rms_rcv = rms_line_fit(rcv_pts) if rcv_pts.size else 0.0
                        else:
                            rms_drv = 0.0
                            rms_rcv = 0.0

                        # EMA a kijelzéshez
                        rms_drv_disp = rms_drv_s.update(rms_drv)
                        rms_rcv_disp = rms_rcv_s.update(rms_rcv)

                        # HUD: két sorral lejjebb írjuk ki
                        # Szín kiválasztás RMS-hez
                        def pick_color_rms(val: float):
                            if val >= HANDLE_RMS_ALERT: return (0, 0, 255)
                            if val >= HANDLE_RMS_WARN:  return (0, 165, 255)
                            return (0, 200, 0)
                        color_rms_drv = pick_color_rms(rms_drv_disp)
                        color_rms_rcv = pick_color_rms(rms_rcv_disp)

                        # HUD: két sorral lejjebb írjuk ki, dot-tal
                        cv2.circle(image, (20, hud_y + 44 - 8), 6, color_rms_drv, -1)
                        cv2.putText(
                            image,
                            f"Handle straightness RMS (Drive): {rms_drv_disp:.3f}",
                            (35, hud_y + 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_rms_drv,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.circle(image, (20, hud_y + 66 - 8), 6, color_rms_rcv, -1)
                        cv2.putText(
                            image,
                            f"Handle straightness RMS (Recovery): {rms_rcv_disp:.3f}",
                            (35, hud_y + 66),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_rms_rcv,
                            2,
                            cv2.LINE_AA,
                        )

                        # Catch dwell: eltávolítva
                    # --------------------------------------------------------------------------------

            else:
                cv2.putText(image, "Jelenleg a bal oldal támogatott", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Mediapipe Feed", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    export_csv(fsm)


# ----------------------------- CLI -------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ergo stroke-analízis")
    p.add_argument("--video", required=False, help="Videófájl elérési útja")
    p.add_argument("--side", choices=["bal","jobb"], default="bal", help="Nézett oldal")
    return p.parse_args()

def main():
    args = parse_args()
    video = args.video
    if not video:
        # Alapértelmezett tesztvideó választása a repo-ból, ha nincs megadva --video
        import os
        candidates = [
           # os.path.join(os.path.dirname(__file__), "Test_01.mp4"),
           # os.path.join(os.path.dirname(__file__), "Test_02.mp4"),
           # os.path.join(os.path.dirname(__file__), "Test_03.mp4"),
           # os.path.join(os.path.dirname(__file__), "Test_04.mp4"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "Videos", "Test_05.mp4"),
            # os.path.join(os.path.dirname(os.path.dirname(__file__)), "Működő verzió", "Test_06.mp4"),
        ]
        video = next((p for p in candidates if os.path.exists(p)), None)
        if not video:
            print("Nem található alapértelmezett tesztvideó. Add meg a --video paramétert.")
            return
        print(f"Alapértelmezett videó: {video}")
    video_feldolgozas(video_path=video, side=args.side)

if __name__ == "__main__":
    main()

