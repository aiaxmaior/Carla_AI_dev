# Tools/HUDTopBar.py
import pygame
from typing import Optional, Tuple

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _risk_color01(v: Optional[float]) -> Tuple[int,int,int]:
    """v in [0,1], green->yellow->red. None -> neutral."""
    if v is None: return (200, 200, 210)
    v = _clamp(v, 0.0, 1.0)
    # piecewise: 0..0.5 green->yellow, 0.5..1 yellow->red
    if v <= 0.5:
        t = v/0.5
        r = int( 40 + t*(230-40) )
        g = int(200 + t*(200-200))  # stays ~200, slight tweak
        b = 60
    else:
        t = (v-0.5)/0.5
        r = 230
        g = int(200 - t*(180))  # 200->20
        b = 60
    return (r,g,b)

class HUDTopBar:
    def __init__(self, screen_size: Tuple[int,int], font_name: Optional[str]=None):
        w, h = screen_size
        self.w, self.h = w, h
        self.scale = min(w, h) / 1080.0
        self.px = lambda v: int(round(v * self.scale))

        self.margin = self.px(16)
        self.bar_h  = self.px(120)      # top bar height (120 @ 1080p)
        self.radius = self.px(14)
        self.pad    = self.px(18)
        self.shadow_offset = self.px(3)

        # Fonts (fallback to default if the named one isn't present)
        self.font_big = pygame.font.SysFont(font_name or "Inter", self.px(64), bold=True)
        self.font_mid = pygame.font.SysFont(font_name or "Inter", self.px(28), bold=True)
        self.font_sm  = pygame.font.SysFont(font_name or "Inter", self.px(22))

        # Pre-made surfaces
        self.bg_rect = pygame.Rect(self.margin, self.margin, self.w - 2*self.margin, self.bar_h)
        self.bg_surf = pygame.Surface((self.bg_rect.w, self.bg_rect.h), pygame.SRCALPHA)
        self.shadow  = pygame.Surface((self.bg_rect.w, self.bg_rect.h), pygame.SRCALPHA)

        self._redraw_bg()

    def _redraw_bg(self):
        self.bg_surf.fill((0,0,0,0))
        self.shadow.fill((0,0,0,0))
        # shadow
        pygame.draw.rect(self.shadow, (0,0,0,110), self.shadow.get_rect(), border_radius=self.radius)
        # background
        pygame.draw.rect(self.bg_surf, (15,15,22,215), self.bg_surf.get_rect(), border_radius=self.radius)

    def resize(self, screen_size: Tuple[int,int]):
        self.__init__(screen_size)

    def draw(
        self,
        screen: pygame.Surface,
        *,
        speed_kmh: Optional[float],
        driver_score: Optional[float],     # 0..100 (your MVD normalized)
        lane_keeping: Optional[float],     # 0..100
        collision_risk: Optional[float],   # 0..1   (probability or normalized risk)
        notification: Optional[str] = None,
        units: str = "km/h",
    ):
        # Place containers
        screen.blit(self.shadow, (self.bg_rect.x, self.bg_rect.y + self.shadow_offset))
        screen.blit(self.bg_surf, (self.bg_rect.x, self.bg_rect.y))

        left_x = self.bg_rect.x + self.pad
        top_y  = self.bg_rect.y + self.pad
        mid_y  = self.bg_rect.y + self.bg_rect.h // 2

        # --- Left: Speed (prominent)
        show_units = "mph" if units.lower().startswith("m") else "km/h"
        spd_txt = "--" if speed_kmh is None else str(int(round(speed_kmh)))
        spd_render = self.font_big.render(spd_txt, True, (235,235,245))
        units_render = self.font_sm.render(show_units, True, (180,180,190))

        screen.blit(spd_render, (left_x, top_y))
        screen.blit(units_render, (left_x + spd_render.get_width() + self.px(8), top_y + spd_render.get_height() - units_render.get_height() - self.px(6)))

        # --- Center: three metrics (Driver Score, Lane Keeping, Collision Risk)
        center_w = self.bg_rect.w - 2*self.pad
        cell_w   = center_w // 3
        base_x   = self.bg_rect.x + self.pad

        def draw_metric(ix, label, value_txt, color=(220,220,230)):
            cx = base_x + ix*cell_w
            label_r = self.font_sm.render(label, True, (170,170,180))
            val_r   = self.font_mid.render(value_txt, True, color)
            # center inside the cell
            lx = cx + (cell_w - label_r.get_width())//2
            vx = cx + (cell_w - val_r.get_width())//2
            screen.blit(label_r, (lx, mid_y - label_r.get_height() - self.px(6)))
            screen.blit(val_r,   (vx, mid_y + self.px(6)))

        # Driver Score (MVD) — keep wording simple
        if driver_score is None:
            ds_val = "--"
        else:
            ds_val = f"{int(round(driver_score))}/100"
        draw_metric(0, "Driver Score", ds_val)

        # Lane Keeping
        lk_val = "--" if lane_keeping is None else f"{int(round(lane_keeping))}/100"
        draw_metric(1, "Lane Keeping", lk_val)

        # Collision Risk — color-coded
        if collision_risk is None:
            cr_val = "--"
            cr_col = (220,220,230)
        else:
            # if you store risk as %, convert to 0..1 outside or clamp here
            cr = _clamp(collision_risk, 0.0, 1.0)
            cr_val = f"{int(round(cr*100))}%"
            cr_col = _risk_color01(cr)
        draw_metric(2, "Collision Risk", cr_val, cr_col)

        # --- Right: Notification pill (optional)
        if notification:
            pill_pad_x = self.px(14); pill_pad_y = self.px(8)
            txt = self.font_sm.render(notification, True, (245,245,250))
            pill_w = txt.get_width() + 2*pill_pad_x
            pill_h = txt.get_height() + 2*pill_pad_y

            pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
            pygame.draw.rect(pill_surf, (35,35,42,230), pill_surf.get_rect(), border_radius=self.px(20))
            pygame.draw.rect(pill_surf, (90,90,110,180), pill_surf.get_rect(), width=self.px(2), border_radius=self.px(20))

            # right aligned inside bar
            px = self.bg_rect.x + self.bg_rect.w - self.pad - pill_w
            py = self.bg_rect.y + self.px(16)
            screen.blit(pill_surf, (px, py))
            screen.blit(txt, (px + pill_pad_x, py + pill_pad_y))
