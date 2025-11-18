# PreWelcomeSelect.py
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Pre-startup UI (resolution, traffic settings)
# [ ] | Hot-path functions: None (runs BEFORE main loop)
# [ ] |- Heavy allocs in hot path? N/A (startup only)
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [X] | Graphics here? YES - full pygame UI (but pre-simulation)
# [ ] | Data produced (tick schema?): Config choices only
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] NOT in hot path - runs before simulation starts
# 2. [PERF_OK] UI rendering acceptable (one-time startup)
# ============================================================================

import pygame
import os
import sys

RES_OPTS = [(1280,720), (1600,900), (1920,1080), (2560,1440)]
CAR_OPTS = [0, 5, 10, 15, 20]
PED_OPTS = [0, 10, 15, 20, 30, 50]
DEV_OPTS = [True, False]

WIDTH, HEIGHT = 1500, 800
LOGO_PATHS = ["./assets/logo.png", "./assets/images/logo.png", "./images/Logo_product.png"]

idx_res  = RES_OPTS.index((1920,1080))
idx_cars = CAR_OPTS.index(10)
idx_peds = PED_OPTS.index(10)
idx_dev = DEV_OPTS.index(False)

# ---------- visuals ----------
BG       = (14, 15, 18)
HEADERBG = (20, 21, 25)
CARD     = (28, 30, 36)
TEXT     = (235, 236, 240)
SUBTEXT  = (185, 186, 190)
BORDER   = (60, 64, 72)
FOCUS    = (74, 124, 230)   # border when focused

def _load_logo(max_w=800, max_h=210):
    for p in LOGO_PATHS:
        if os.path.isfile(p):
            try:
                img = pygame.image.load(p).convert_alpha()
                w, h = img.get_width(), img.get_height()
                scale = min(max_w / w, max_h / h, 1.0)
                if scale < 1.0:
                    img = pygame.transform.smoothscale(img, (int(w*scale), int(h*scale)))
                return img
            except Exception:
                pass
    return None

def _rounded_rect(surface, color, rect, radius=12, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)

def _shadow(surface, rect, radius=12, offset=(0, 6)):
    """Cheap soft shadow: a few translucent layers behind the rect."""
    ox, oy = offset
    for i, alpha in enumerate((80, 50, 25)):
        grow = 6 + i * 4
        rr = pygame.Rect(rect.x - grow + ox, rect.y - grow + oy,
                         rect.w + 2*grow, rect.h + 2*grow)
        col = (0, 0, 0, alpha)
        sh = pygame.Surface((rr.w, rr.h), pygame.SRCALPHA)
        _rounded_rect(sh, col, pygame.Rect(0, 0, rr.w, rr.h), radius=radius+4+i)
        surface.blit(sh, (rr.x, rr.y))

def _card(surface, rect, focused=False, border=True):
    _shadow(surface, rect)
    _rounded_rect(surface, CARD, rect, radius=14, width=0)
    if border:
        _rounded_rect(surface, FOCUS if focused else BORDER, rect, radius=14, width=2)

def pre_welcome_select(default_res=(1920,1080), default_cars=10, default_peds=10):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quick Setup")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    small = pygame.font.SysFont(None, 22)

    # --- defaults -> starting indices (first visible) ---
    def _res_idx(val):
        if val in RES_OPTS: return RES_OPTS.index(val)
        area = val[0]*val[1]
        return min(range(len(RES_OPTS)), key=lambda i: abs(RES_OPTS[i][0]*RES_OPTS[i][1] - area))
    def _near(options, v): return min(range(len(options)), key=lambda i: abs(options[i]-v))

    idx_res  = _res_idx(default_res)
    idx_cars = _near(CAR_OPTS, default_cars)
    idx_peds = _near(PED_OPTS, default_peds)
    idx_dev = DEV_OPTS.index(False)
    focus = 0  # 0=Resolution, 1=Cars, 2=Peds

    logo = _load_logo()

    def draw():
        screen.fill(BG)
        # header
        pygame.draw.rect(screen, HEADERBG, (0, 0, WIDTH, 260))
        if logo:
            lw, lh = logo.get_width(), logo.get_height()
            screen.blit(logo, ((WIDTH - lw)//2, 12))
        else:
            title = font.render("Quick Setup", True, TEXT)
            screen.blit(title, ((WIDTH - title.get_width())//2, 42))

        # rows area
        y0, row_h, gap = 240, 68, 18
        left_x, right_x = 60, WIDTH - 60
        items = [
            ("Resolution", f"{RES_OPTS[idx_res][0]}×{RES_OPTS[idx_res][1]}"),
            ("Cars",        str(CAR_OPTS[idx_cars])),
            ("Peds",        str(PED_OPTS[idx_peds])),
            ("Developer Mode", "On" if DEV_OPTS[idx_dev] else "Off")
        ]
        for i, (k, v) in enumerate(items):
            y = y0 + i * (row_h + gap)
            rect = pygame.Rect(40, y - 10, WIDTH - 80, row_h)
            _card(screen, rect, focused=(i == focus))

            key = font.render(k, True, TEXT)
            val = font.render(v, True, TEXT)
            screen.blit(key, (left_x, y))
            screen.blit(val, (right_x - val.get_width(), y))

        hint = small.render("↑/↓ move  •  ←/→ change  •  Enter confirm  •  Esc quit", True, SUBTEXT)
        screen.blit(hint, ((WIDTH - hint.get_width())//2, HEIGHT - 40))
        pygame.display.flip()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); return None
                if e.key in (pygame.K_UP, pygame.K_w):
                    focus = (focus - 1) % 4  # was % 3
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    focus = (focus + 1) % 4  # was % 3
                elif e.key in (pygame.K_LEFT, pygame.K_a):
                    if focus == 0: idx_res  = (idx_res  - 1) % len(RES_OPTS)
                    if focus == 1: idx_cars = (idx_cars - 1) % len(CAR_OPTS)
                    if focus == 2: idx_peds = (idx_peds - 1) % len(PED_OPTS)
                    if focus == 3: idx_dev  = (idx_dev  - 1) % len(DEV_OPTS)
                elif e.key in (pygame.K_RIGHT, pygame.K_d):
                    if focus == 0: idx_res  = (idx_res  + 1) % len(RES_OPTS)
                    if focus == 1: idx_cars = (idx_cars + 1) % len(CAR_OPTS)
                    if focus == 2: idx_peds = (idx_peds + 1) % len(PED_OPTS)
                    if focus == 3: idx_dev  = (idx_dev  + 1) % len(DEV_OPTS)
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    sel = (RES_OPTS[idx_res], PED_OPTS[idx_peds], CAR_OPTS[idx_cars], DEV_OPTS[idx_dev])
                    pygame.quit()
                    return sel  # removed sys.exit()
        draw()
        clock.tick(60)

if __name__ == "__main__":
    out = pre_welcome_select()
    if out:
        (w,h), n_peds, n_cars, dev_bool = out
        print(f"Selected: {w}x{h}, peds={n_peds}, cars={n_cars}")
        print(f"Dev mode: {dev_bool}")