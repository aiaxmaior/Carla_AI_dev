# dash.py
import os, re, subprocess, pygame, sys

DASH_OUT = sys.argv[1] if len(sys.argv) > 1 else "HDMI-0"  # pass output name if needed

def rect_for(output):
    xr = subprocess.check_output(['xrandr','--query']).decode()
    m = re.search(rf'^{output}\s+connected.*?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)', xr, re.M)
    if not m: raise SystemExit(f"{output} not found/connected")
    w,h,x,y = map(int, m.groups()); return x,y,w,h

x,y,w,h = rect_for(DASH_OUT)
win_w, win_h = 1920,1080
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y + max(0,(h - win_h)//2)}"

pygame.init()
screen = pygame.display.set_mode((win_w,win_h), pygame.NOFRAME)
clock = pygame.time.Clock()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
    # draw your dash here…
    pygame.display.flip()
    clock.tick(60)
 w