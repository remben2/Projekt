# Központosított konstansok

# Drive cue hiszterézis (térd szögsebesség)
ENTER_THR = 40.0   # deg/s
EXIT_THR  = 20.0   # deg/s
HOLD_N    = 3      # egymást követő frame

# SPM időkapu (s / stroke)
SPM_MIN_S = 0.3    # 200 spm felett dobjuk
SPM_MAX_S = 5.0    # 12 spm alatt dobjuk

# Törzs max frissítés derivált küszöb
TRUNK_DERIV_THR = 2.0  # deg/s

# EMA simítók
ALPHA_ELBOW = 0.35
ALPHA_KNEE  = 0.25
ALPHA_HIP   = 0.25
ALPHA_TRUNK = 0.50
ALPHA_BACK  = 0.40
BACK_HOLD_S = 0.5  # másodperc – utolsó jó hátkontúrt ennyi ideig tartjuk
BACK_SIDE_HOLD_N = 6  # ennyi egymást követő frame kell oldalváltáshoz (hysteresis)

# Hátgörbe/bend HUD küszöbök
CURVE_WARN = 0.18
CURVE_ALERT = 0.25
BEND_WARN = 145.0   # fok
BEND_ALERT = 135.0  # fok
