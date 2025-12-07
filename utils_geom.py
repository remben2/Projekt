from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2


def calculate_angle(a, b, c) -> float:
    """Szög a-b-c pontokból (fok, 0..180)."""
    ax, ay = a; bx, by = b; cx, cy = c
    radians = np.arctan2(cy - by, cx - bx) - np.arctan2(ay - by, ax - bx)
    angle = abs(np.degrees(radians))
    return float(360.0 - angle if angle > 180.0 else angle)


def trunk_angle_deg(shoulder_xy, hip_xy) -> float:
    """Törzs dőlése a függőlegestől: 0°=függőleges, 90°=vízszintes (irányfüggetlen)."""
    v = np.array(shoulder_xy, dtype=float) - np.array(hip_xy, dtype=float)
    vx, vy = float(v[0]), float(v[1])
    norm = np.hypot(vx, vy) + 1e-9
    cosang = np.clip(abs(-vy) / norm, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def get_xy(lm, idx, min_vis: float = 0.5):
    pt = lm[idx]
    vis = getattr(pt, "visibility", 1.0)
    if vis is not None and vis < min_vis:
        return None
    if pt.x is None or pt.y is None:
        return None
    return [float(pt.x), float(pt.y)]


def back_contour_from_mask(
    mask: np.ndarray,
    shoulder_xy: Tuple[float, float],
    hip_xy: Tuple[float, float],
    n_samples: int = 40,
    thresh: float = 0.5,
    max_radius: int = 30,
    prefer_sign: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Nagyon egyszerű kontúrszedés a váll–csípő között a szegmentációs maszkból.
    Visszaad: képpont koordináták listája (x,y), amely a hát közelítő görbéje.

    Lépések:
    - váll→csípő irányú vektor, majd ortogonális normalizált vektor (hát felületére merőleges sugarak)
    - minden mintapontnál egy rövid keresés a maszk szélén (háttér→test átmenet)
    - ez egy gyors, zajos közelítés; később stabilizálható
    """
    if mask is None or mask.ndim != 2:
        return []
    H, W = mask.shape[:2]
    sx = int(round(shoulder_xy[0] * W)); sy = int(round(shoulder_xy[1] * H))
    hx = int(round(hip_xy[0] * W));      hy = int(round(hip_xy[1] * H))

    # váll->csípő vektor és egységvektorok
    vx, vy = hx - sx, hy - sy
    L = float(np.hypot(vx, vy)) + 1e-6
    ux, uy = vx / L, vy / L
    # ortogonális egységvektor (két irányban is keresünk)
    nx, ny = -uy, ux

    pts: List[Tuple[int, int]] = []
    y_min = min(sy, hy) - 3
    y_max = max(sy, hy) + 3
    # végigmegyünk a gerincvonal mentén és keresünk test-szél pontokat
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        cx = sx + t * vx
        cy = sy + t * vy
        # Keresés mindkét normál irányban, az első kinti pixelig
        def first_out(sign: int) -> Tuple[int, int] | None:
            for r in range(0, max_radius):
                x = int(round(cx + sign * r * nx))
                y = int(round(cy + sign * r * ny))
                if x < 0 or x >= W or y < 0 or y >= H:
                    return None
                if mask[y, x] <= thresh:
                    # lépjünk vissza 1 pixelt, hogy a test peremén maradjunk
                    bx = int(round(cx + sign * max(r - 1, 0) * nx))
                    by = int(round(cy + sign * max(r - 1, 0) * ny))
                    return (bx, by)
            return None

        # preferált oldal – csak azon az oldalon keresünk, ha be van állítva
        p_plus = p_minus = None
        if prefer_sign is not None:
            cand = first_out(prefer_sign)
            if cand is not None:
                px, py = cand
                if y_min <= py <= y_max:
                    pts.append((px, py))
                    continue
            # ha nincs találat a preferált oldalon, utolsó esélyként másik oldal
            cand = first_out(-prefer_sign)
            if cand is not None:
                px, py = cand
                if y_min <= py <= y_max:
                    pts.append((px, py))
                    continue
            # végső fallback: középpont
            pts.append((int(round(cx)), int(round(cy))))
            continue

        # nincs preferált oldal – vizsgáljuk mindkettőt és válasszuk a távolabbit, de tartsuk a y-sávot
        p_plus  = first_out(+1)
        p_minus = first_out(-1)

        # Válasszuk a távolabbit a gerincvonaltól (általában ez a külső hátszegély)
        def dist_sq(pt: Tuple[int, int] | None) -> float:
            if pt is None:
                return -1.0
            dx = float(pt[0]) - cx
            dy = float(pt[1]) - cy
            return dx * dx + dy * dy

        d_plus = dist_sq(p_plus)
        d_minus = dist_sq(p_minus)

        if d_plus < 0 and d_minus < 0:
            best = (int(round(cx)), int(round(cy)))
        else:
            best = p_plus if d_plus >= d_minus else p_minus  # válaszd a nagyobbat
            if best is None:
                best = (int(round(cx)), int(round(cy)))
        # y-szűrés a váll–csípő sávra
        if y_min <= best[1] <= y_max:
            pts.append(best)

    return pts


def back_curvature_metric(
    contour: List[Tuple[int, int]],
    shoulder_xy: Tuple[float, float],
    hip_xy: Tuple[float, float],
    image_shape: Tuple[int, int, int] | Tuple[int, int],
) -> Tuple[float, float]:
    """
    Egyszerű görbületi mérőszám a kontúrból:
      - Max. merőleges eltérés a váll–csípő egyenestől (pixelben)
      - Normalizált arány: max_dev / |váll-hip| (0..1 tartományba várhatóan 0..0.4)

    Visszatér: (norm_ratio, max_dev_px)
    """
    if not contour:
        return 0.0, 0.0
    if len(image_shape) == 3:
        H, W = image_shape[:2]
    else:
        H, W = image_shape  # type: ignore

    sx = float(shoulder_xy[0] * W); sy = float(shoulder_xy[1] * H)
    hx = float(hip_xy[0] * W);      hy = float(hip_xy[1] * H)

    vx = hx - sx; vy = hy - sy
    L = float(np.hypot(vx, vy))
    if L < 1e-3:
        return 0.0, 0.0
    ux, uy = vx / L, vy / L
    # unit normal (válasszunk egy fix orientációt)
    nx, ny = -uy, ux

    max_dev = 0.0
    for (px, py) in contour:
        dx = float(px) - sx
        dy = float(py) - sy
        # vetítés a gerincvonalra, csak a [0, L] szakaszban mérünk
        t = dx * ux + dy * uy
        if t < 0 or t > L:
            continue
        # merőleges komponens nagysága a normál mentén
        dev = abs(dx * nx + dy * ny)
        if dev > max_dev:
            max_dev = dev

    ratio = float(max_dev / L) if L > 1e-6 else 0.0
    return ratio, max_dev


def back_bend_angle(
    contour: List[Tuple[int, int]],
    shoulder_xy: Tuple[float, float],
    hip_xy: Tuple[float, float],
    image_shape: Tuple[int, int, int] | Tuple[int, int],
) -> float:
    """
    Visszaad egy "derék-szerű" szöget a hát görbéjéből.
    Módszer: kiválasztjuk a max. eltérésű pontot P a kontúron, majd
      angle = ∠(S, P, H) fokban (klasszikus hárompontos szög),
    ahol S=v&aacute;ll, H=csípő (pixel koordináták).
    Megjegyzés: ez egy vizuális proxy; később finomítható spline-illesztéssel.
    """
    if not contour:
        return 0.0
    if len(image_shape) == 3:
        H, W = image_shape[:2]
    else:
        H, W = image_shape  # type: ignore

    # konvertáljuk S, H pixelbe
    sx = float(shoulder_xy[0] * W); sy = float(shoulder_xy[1] * H)
    hx = float(hip_xy[0] * W);      hy = float(hip_xy[1] * H)

    # keressük meg a max. eltérésű pontot
    vx = hx - sx; vy = hy - sy
    L = float(np.hypot(vx, vy))
    if L < 1e-3:
        return 0.0
    ux, uy = vx / L, vy / L
    nx, ny = -uy, ux

    def dev_on_normal(px: float, py: float) -> float:
        dx = px - sx; dy = py - sy
        t = dx * ux + dy * uy
        if t < 0 or t > L:
            return -1.0
        return abs(dx * nx + dy * ny)

    max_i = -1
    max_d = -1.0
    for i, (px, py) in enumerate(contour):
        d = dev_on_normal(float(px), float(py))
        if d > max_d:
            max_d = d
            max_i = i
    if max_i < 0:
        return 0.0

    P = contour[max_i]
    # hárompontos szög S-P-H
    a = np.array([sx, sy], dtype=float)
    b = np.array([float(P[0]), float(P[1])], dtype=float)
    c = np.array([hx, hy], dtype=float)
    ab = a - b
    cb = c - b
    na = np.linalg.norm(ab) + 1e-9
    nc = np.linalg.norm(cb) + 1e-9
    cosang = np.clip(np.dot(ab, cb) / (na * nc), -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cosang)))
    return ang


def back_contour_from_edges(
    mask: np.ndarray,
    shoulder_xy: Tuple[float, float],
    hip_xy: Tuple[float, float],
    prefer_sign: int | None,
    n_keep: int = 50,
    thresh: float = 0.5,
    t_margin: float = 0.05,
) -> List[Tuple[int, int]]:
    """
    Kontúralapú hát-vonal: a bináris maszk külső kontúrjából kiválasztjuk azokat
    a pontokat, amelyek a váll–csípő egyenes hát felőli oldalára esnek, és
    a szakasz vetített tartományán belül vannak. A kiválasztott pontokat t szerint
    rendezzük és ritkítjuk.
    """
    if mask is None or mask.ndim != 2:
        return []
    H, W = mask.shape[:2]
    sx = float(shoulder_xy[0] * W); sy = float(shoulder_xy[1] * H)
    hx = float(hip_xy[0] * W);      hy = float(hip_xy[1] * H)

    vx = hx - sx; vy = hy - sy
    L = float(np.hypot(vx, vy))
    if L < 5.0:
        return []
    ux, uy = vx / L, vy / L
    nx, ny = -uy, ux  # normál

    # küszöbölés és kontúrok
    bin_img = (mask > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts: List[Tuple[int, int, float]] = []  # (x,y,t)
    t_lo = -t_margin * L
    t_hi = (1.0 + t_margin) * L
    for cnt in contours:
        if cnt is None or len(cnt) == 0:
            continue
        arr = cnt.reshape(-1, 2)
        for px, py in arr:
            dx = float(px) - sx; dy = float(py) - sy
            t = dx * ux + dy * uy
            if t < t_lo or t > t_hi:
                continue
            # preferált oldal szűrés
            if prefer_sign is not None:
                side = dx * nx + dy * ny
                if prefer_sign * side <= 0:
                    continue
            pts.append((int(px), int(py), t))

    if not pts:
        return []
    # rendezés t szerint, majd ritkítás egyenletes lépéssel
    pts.sort(key=lambda p: p[2])
    if len(pts) > n_keep:
        step = max(1, len(pts) // n_keep)
        pts = pts[::step]
    return [(p[0], p[1]) for p in pts]
