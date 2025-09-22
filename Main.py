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
from utils_geom import calculate_angle, trunk_angle_deg, get_xy
from utils_filters import EMASmoother, WindowDerivative, WindowPeak
from stroke_fsm import StrokeFSM
from constants import (
    ENTER_THR, EXIT_THR, HOLD_N,
    SPM_MIN_S, SPM_MAX_S,
    TRUNK_DERIV_THR,
    ALPHA_ELBOW, ALPHA_KNEE, ALPHA_HIP, ALPHA_TRUNK,
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
            "score_total","score_ratio","score_spm","score_posture"
        ])
        w.writerows(fsm.log)
    print(f"Mentve: {path}   sorok: {len(fsm.log)}")

def video_feldolgozas(video_path: str, side: str = "bal"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / float(fps)

        elbow_s = EMASmoother(ALPHA_ELBOW)
        knee_s  = EMASmoother(ALPHA_KNEE)
        hip_s   = EMASmoother(ALPHA_HIP)
        trunk_s = EMASmoother(ALPHA_TRUNK)
        knee_der = WindowDerivative(window=3, dt=dt)

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
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "teszt","Videos", "Test_05.mp4"),
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

