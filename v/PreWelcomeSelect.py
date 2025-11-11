# PreWelcomeSelect.py
import pygame

RES_OPTS = [(1280,720), (1600,900), (1920,1080), (2560,1440),(3840,2160)]
CAR_OPTS = [0, 5, 10, 25, 50, 100, 150]
PED_OPTS = [0, 5, 25, 50, 100, 150, 200]

def pre_welcome_select():
    pygame.init()
    screen = pygame.display.set_mode((800, 420))
    pygame.display.set_caption("Quick Setup")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 34)
    small = pygame.font.SysFont(None, 24)

    rows = ["Resolution", "Cars", "Peds"]
    idx_res, idx_cars, idx_peds = 2, 2, 2  # defaults (1920x1080, 25 cars, 50 peds)
    focus = 0

    def draw():
        screen.fill((18,18,18))
        title = font.render("Quick Setup", True, (220,220,220))
        screen.blit(title, (30, 20))

        y0 = 100
        items = [
            ("Resolution", f"{RES_OPTS[idx_res][0]}×{RES_OPTS[idx_res][1]}"),
            ("Cars", str(CAR_OPTS[idx_cars])),
            ("Peds", str(PED_OPTS[idx_peds])),
        ]
        for i,(k,v) in enumerate(items):
            y = y0 + i*70
            # focus highlight bar
            if i == focus:
                pygame.draw.rect(screen, (50,90,170), (20, y-10, 760, 50), border_radius=10)
            key = font.render(k, True, (245,245,245))
            val = font.render(v, True, (245,245,245))
            screen.blit(key, (40, y))
            screen.blit(val, (560, y))

        hint = small.render("↑/↓ move  •  ←/→ change  •  Enter confirm  •  Esc quit", True, (180,180,180))
        screen.blit(hint, (30, 360))
        pygame.display.flip()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); return None
                if e.key in (pygame.K_UP, pygame.K_w):
                    focus = (focus - 1) % 3
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    focus = (focus + 1) % 3
                elif e.key in (pygame.K_LEFT, pygame.K_a):
                    if focus == 0: idx_res  = (idx_res  - 1) % len(RES_OPTS)
                    if focus == 1: idx_cars = (idx_cars - 1) % len(CAR_OPTS)
                    if focus == 2: idx_peds = (idx_peds - 1) % len(PED_OPTS)
                elif e.key in (pygame.K_RIGHT, pygame.K_d):
                    if focus == 0: idx_res  = (idx_res  + 1) % len(RES_OPTS)
                    if focus == 1: idx_cars = (idx_cars + 1) % len(CAR_OPTS)
                    if focus == 2: idx_peds = (idx_peds + 1) % len(PED_OPTS)
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    sel = (RES_OPTS[idx_res], PED_OPTS[idx_peds], CAR_OPTS[idx_cars], False)  # Add dev_mode=False
                    pygame.quit()
                    return sel
        draw()
        clock.tick(60)

if __name__ == "__main__":
    out = pre_welcome_select()
    if out:
        (w,h), n_peds, n_cars = out
        print(f"Selected: {w}x{h}, peds={n_peds}, cars={n_cars}")
