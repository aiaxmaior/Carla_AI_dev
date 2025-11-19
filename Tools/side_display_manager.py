
# side_display_manager.py
# -------------------------------------------------------------
# Arrange: 49" super-ultrawide as primary + two side displays.
# If only one side display is present, create a *virtual* side
# region on the ultrawide so you can still place a window there.
#
# Also exposes helpers to compute window rectangles and to spawn
# borderless demo windows using pygame (optional).
# -------------------------------------------------------------
import re, subprocess, shlex, sys, argparse, json

def run(cmd, check=False):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(shlex.quote, cmd))}\n{p.stderr}")
    return p

def xrandr_query():
    return run(['xrandr','--query']).stdout

def parse_outputs(qtext):
    # returns: dict name -> {w,h,x,y,connected,primary,modes: [..]}
    blocks = re.split(r'(?=^[A-Za-z0-9-]+\s+(?:connected|disconnected))', qtext, flags=re.M)
    out = {}
    for b in blocks:
        b = b.strip()
        if not b: continue
        mname = re.match(r'^([A-Za-z0-9-]+)\s+(connected|disconnected)(?:\s+primary)?\s*', b)
        if not mname: continue
        name, state = mname.group(1), mname.group(2)
        connected = (state == 'connected')
        primary = ' connected primary ' in (b[:80] + ' ')
        mcur = re.search(r'(\d{3,5})x(\d{3,5})\+(-?\d+)\+(-?\d+)', b)
        if mcur:
            w, h, x, y = map(int, mcur.groups())
            mode = f"{w}x{h}"
        else:
            w = h = x = y = 0
            mode = None
        modes = []
        for line in b.splitlines()[1:]:
            m = re.match(r'\s*(\d{3,5}x\d{3,5})\b', line)
            if m: modes.append(m.group(1))
        out[name] = dict(name=name, connected=connected, primary=primary, w=w, h=h, x=x, y=y, mode=mode, modes=modes)
    return out

def choose_big(outputs):
    best = None
    for o in outputs.values():
        if not o['connected'] or not o['w'] or not o['h']: continue
        area = o['w']*o['h']
        if not best or area > best['w']*best['h']:
            best = o
    return best['name'] if best else None

def choose_sides(outputs, big_name):
    names = [o['name'] for o in outputs.values() if o['connected'] and o['name'] != big_name]
    def score(n):
        o = outputs[n]
        return (o['w']*o['h'], o['w'])
    names.sort(key=score, reverse=True)
    left = names[0] if len(names) > 0 else None
    right = names[1] if len(names) > 1 else None
    return left, right

def arrange_layout(outputs, big, left=None, right=None, center_vert=True, rate=None):
    cmds = []
    bw, bh = outputs[big]['w'], outputs[big]['h']
    bmode = outputs[big]['mode'] or f"{bw}x{bh}"
    left_w = outputs[left]['w'] if left else 0
    bx = left_w if left else 0
    by = 0
    base = ['xrandr', '--output', big, '--mode', bmode, '--primary', '--pos', f'{bx}x{by}']
    if rate:
        base += ['--rate', str(rate)]
    cmds.append(base)

    if left:
        lw, lh = outputs[left]['w'], outputs[left]['h']
        ly = max(0, (bh - lh)//2) if center_vert else 0
        cmds.append(['xrandr', '--output', left, '--mode', f'{lw}x{lh}', '--pos', f'0x{ly}'])

    if right:
        rw, rh = outputs[right]['w'], outputs[right]['h']
        rx = bx + bw
        ry = max(0, (bh - rh)//2) if center_vert else 0
        cmds.append(['xrandr', '--output', right, '--mode', f'{rw}x{rh}', '--pos', f'{rx}x{ry}'])

    keep = {big}
    if left: keep.add(left)
    if right: keep.add(right)
    for name, o in outputs.items():
        if o['connected'] and name not in keep:
            cmds.append(['xrandr', '--output', name, '--off'])

    for c in cmds:
        run(c)

def compute_side_rects(outputs, big, left=None, right=None, side_size=(1920,1080)):
    bw, bh, bx, by = outputs[big]['w'], outputs[big]['h'], outputs[big]['x'], outputs[big]['y']
    rects = {}

    def center_y(h): return by + max(0, (bh - h)//2)

    if left and outputs[left]['connected']:
        lw, lh, lx, ly = outputs[left]['w'], outputs[left]['h'], outputs[left]['x'], outputs[left]['y']
        rects['left'] = dict(x=lx, y=center_y(lh), w=lw, h=lh, is_virtual=False, output=left)
    else:
        vw, vh = side_size
        rects['left'] = dict(x=bx + max(0, (bw//2 - vw)//2), y=center_y(vh), w=vw, h=vh, is_virtual=True, output=big)

    if right and outputs[right]['connected']:
        rw, rh, rx, ry = outputs[right]['w'], outputs[right]['h'], outputs[right]['x'], outputs[right]['y']
        rects['right'] = dict(x=rx, y=center_y(rh), w=rw, h=rh, is_virtual=False, output=right)
    else:
        vw, vh = side_size
        rects['right'] = dict(x=bx + bw - vw - max(0, (bw//2 - vw)//2), y=center_y(vh), w=vw, h=vh, is_virtual=True, output=big)

    return rects

def demo_window(which, rects, color=(20,200,120)):
    import pygame, os
    r = rects[which]
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{r['x']},{r['y']}"
    pygame.init()
    screen = pygame.display.set_mode((r['w'], r['h']), pygame.NOFRAME)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Inter', 24, bold=True)
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        screen.fill((18,18,24))
        pygame.draw.rect(screen, color, (8,8,r['w']-16, r['h']-16), width=4, border_radius=12)
        txt = font.render(f"{which.upper()}  {'VIRTUAL' if r['is_virtual'] else 'PHYSICAL'}", True, (230,230,240))
        screen.blit(txt, (20,20))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Apply xrandr layout (big + two sides)')
    ap.add_argument('--print-rects', action='store_true', help='Print computed side rects JSON')
    ap.add_argument('--demo', action='store_true', help='Spawn a demo window (requires pygame)')
    ap.add_argument('--side', choices=['left','right'], default='left')
    ap.add_argument('--big', help='Big output name (auto if omitted)')
    ap.add_argument('--left', help='Left output name (optional)')
    ap.add_argument('--right', help='Right output name (optional)')
    ap.add_argument('--no-center-vert', action='store_true', help='Do not vertically center sides')
    ap.add_argument('--rate', type=int, help='Force refresh rate on big (e.g., 120)')
    args = ap.parse_args()

    q = xrandr_query()
    outputs = parse_outputs(q)
    big = args.big or choose_big(outputs)
    if not big:
        print('No big display found.'); sys.exit(2)

    left = args.left
    right = args.right
    if not left or not right:
        auto_left, auto_right = choose_sides(outputs, big)
        left = left or auto_left
        right = right or auto_right

    if args.apply:
        arrange_layout(outputs, big, left, right, center_vert=not args.no_center_vert, rate=args.rate)

    rects = compute_side_rects(outputs, big, left, right)
    if args.print-rects:
        print(json.dumps(rects, indent=2))

    if args.demo:
        demo_window(args.side, rects)

if __name__ == '__main__':
    main()
