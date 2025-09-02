import pygame
import sys
import os
import logging
from VehicleLibrary import VehicleLibrary
from VehicleLibrary import MapLibrary

class TitleScreen(object):
    
    def __init__ (self, display,client,args):
        self._display = display
        self._client = client
        self._H = self._display.get_height()
        self._W = self._display.get_width()
        self._args = args
        self._default_id = 'ford_e450_super_duty'
        self.center_x = None

        font_path = os.path.join(
            self._args.carla_root, "CarlaUE4", "Content", "Carla", "Fonts",
            "tt-supermolot-neue-trl.bd-it.ttf",
        )

        self._font_title = pygame.font.Font(font_path, 64)
        self._font_subtitle = pygame.font.Font(font_path, 32)
        self._font_credits = pygame.font.Font(font_path, 22)
        self._font_prompt = pygame.font.Font(font_path, 42)
        try:
            self._font_title = pygame.font.Font(font_path, 64)
            self._font_subtitle = pygame.font.Font(font_path, 32)
            self._font_credits = pygame.font.Font(font_path, 22)
            self._font_prompt = pygame.font.Font(font_path, 42)
        except pygame.error:
            logging.warning("Custom title font not found, falling back to default.")
            self._font_title = pygame.font.Font(None, 80)
            self._font_subtitle = pygame.font.Font(None, 40)
            self._font_credits = pygame.font.Font(None, 28)
            self._font_prompt = pygame.font.Font(None, 60)

    def select_vehicle_config(self, persistent_keys, center_x, logo_img, default_id="ford_e450_super_duty"):
        default_carla_blueprint = 'vehicle.mercedes.sprinter'
        lib = VehicleLibrary()
        items = lib.list_display_items()   # [{'id': ..., 'name': ...}, ...]
        if not items:
            logging.warning("No vehicle configs found; using default.")
            return default_id

        # Build display/name and id arrays in the same order
        names = [it["name"] for it in items]
        ids   = [it["id"]   for it in items]
        carla_blueprint = [it["carla_blueprint"] for it in items]

        # Pick default index by canonical id (aliases resolved to id)
        try:
            default_canon = lib.resolve_to_id(default_id)
            sel = ids.index(default_canon)
        except Exception:
            sel = 0

        # helpers for mapped presses
        def _is_mapped_press(ev, mapping):
            return (
                ev is not None and
                ev.type == pygame.JOYBUTTONDOWN and
                mapping and
                mapping.get("joy_id") is not None and
                mapping.get("joy_id") == getattr(ev, "instance_id", None) and
                mapping.get("button_id") == getattr(ev, "button", None)
            )

        map_up     = persistent_keys.get("Up")
        map_down   = persistent_keys.get("Down")
        map_enter  = persistent_keys.get("Enter")
        map_escape = persistent_keys.get("Escape")

        clock = pygame.time.Clock()
        hat_cooldown = 0

        while True:
            clock.tick(60)
            if hat_cooldown: hat_cooldown -= 1

            # --- render (centered on your chosen panel via center_x) ---
            self._display.fill((24, 28, 34))
            w, h = self._display.get_size()
            title = self._font_title.render("Select Vehicle Config", True, (200, 220, 255))
            self._display.blit(title, title.get_rect(center=(center_x, int(h * 0.18))))
            start = max(0, sel - 5); end = min(len(names), start + 11)
            y = int(h * 0.30)
            for i in range(start, end):
                pick = (i == sel)
                color = (255, 255, 255) if pick else (180, 190, 200)
                text = ("> " if pick else "  ") + names[i]
                surf = self._font_subtitle.render(text, True, color)
                self._display.blit(surf, surf.get_rect(center=(center_x, y)))
                y += int(self._font_subtitle.get_height() * 1.25)
            hint = "Wheel: Up/Down • Enter to select • Esc to cancel"
            hint_surf = self._font_subtitle.render(hint, True, (160, 170, 180))
            self._display.blit(hint_surf, hint_surf.get_rect(center=(center_x, int(h * 0.9))))
            pygame.display.flip()

            # --- events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return default_id

                # mapped wheel buttons
                if _is_mapped_press(event, map_up):
                    sel = (sel - 1) % len(names); continue
                if _is_mapped_press(event, map_down):
                    sel = (sel + 1) % len(names); continue
                if _is_mapped_press(event, map_enter):
                    return ids[sel], carla_blueprint[sel]
                if _is_mapped_press(event, map_escape):
                    return default_id, default_carla_blueprint

                # keyboard & hats fallback
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        sel = (sel - 1) % len(names)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        sel = (sel + 1) % len(names)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return ids[sel], carla_blueprint[sel]
                    elif event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                        return default_id, default_carla_blueprint

                if event.type == pygame.JOYHATMOTION and hat_cooldown == 0:
                    _, dy = event.value
                    if dy == 1:
                        sel = (sel - 1) % len(names); hat_cooldown = 5
                    elif dy == -1:
                        sel = (sel + 1) % len(names); hat_cooldown = 5


    def select_map(self, persistent_keys, center_x, logo_img, default_id="Town10HD_Opt"):
        """
        Title screen map picker that reads from CARLA's registry.
        - Shows pretty names (e.g., 'Town10 HD Opt')
        - Returns a canonical id like 'Town10HD_Opt' suitable for client.load_world(...)
        """
        import re

        # --- helpers ---
        def _basename(asset: str) -> str:
            # "/Game/Carla/Maps/Town10HD_Opt" -> "Town10HD_Opt"
            return asset.rsplit("/", 1)[-1] if asset else ""

        def _prettify(mid: str) -> str:
            # "Town10HD_Opt" -> "Town10 HD Opt"
            s = mid.replace("_", " ")
            s = s.replace("HD", " HD")
            return s

        def _canon(s: str) -> str:
            # Accept full asset, id-only, or sloppy casing -> normalize to canonical id
            if not s:
                return ""
            base = _basename(s)  # if already id, stays the same
            return base  # keep original casing; CARLA expects exact id like Town10HD_Opt

        def _natural_key(s: str):
            # Natural sort so Town2 < Town10
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

        def _is_mapped_press(ev, mapping):
            return (
                ev is not None and
                ev.type == pygame.JOYBUTTONDOWN and
                mapping and
                mapping.get("joy_id") is not None and
                mapping.get("button_id") is not None and
                # pygame sends instance_id on some builds; fall back to ev.joy if missing
                (mapping.get("joy_id") == getattr(ev, "instance_id", getattr(ev, "joy", None))) and
                (mapping.get("button_id") == getattr(ev, "button", None))
            )

        # --- gather and prepare items ---
        assets = list(self._client.get_available_maps() or [])
        if not assets:
            logging.warning("No maps reported by CARLA; using default id.")
            return _canon(default_id) or "Town01"

        # Make unique (just in case), convert to ids, and sort
        ids = sorted({_basename(a) for a in assets}, key=_natural_key)
        names = [_prettify(mid) for mid in ids]

        # Select default (case-insensitive match on id or asset)
        default_canon = _canon(default_id)
        try:
            sel = ids.index(default_canon)
        except ValueError:
            # try case-insensitive fallback
            lowered = [x.lower() for x in ids]
            try:
                sel = lowered.index(default_canon.lower())
            except ValueError:
                sel = 0

        # --- input mappings ---
        map_up     = persistent_keys.get("Up")
        map_down   = persistent_keys.get("Down")
        map_enter  = persistent_keys.get("Enter")
        map_escape = persistent_keys.get("Escape")

        clock = pygame.time.Clock()
        hat_cooldown = 0

        while True:
            clock.tick(60)
            if hat_cooldown:
                hat_cooldown -= 1

            # --- render ---
            self._display.fill((24, 28, 34))
            if logo_img:
                self._display.blit(logo_img, logo_img.get_rect(center=(center_x, int(self._H * 0.24))))

            title = self._font_title.render("Select Map", True, (200, 220, 255))
            self._display.blit(title, title.get_rect(center=(center_x, int(self._H * 0.14))))

            # windowed list around selection
            start = max(0, sel - 5)
            end   = min(len(names), start + 11)
            y = int(self._H * 0.45)
            for i in range(start, end):
                pick  = (i == sel)
                color = (255, 255, 255) if pick else (180, 190, 200)
                text  = ("> " if pick else "  ") + names[i]
                surf  = self._font_subtitle.render(text, True, color)
                self._display.blit(surf, surf.get_rect(center=(center_x, y)))
                y += int(self._font_subtitle.get_height() * 1.25)

            hint = "Wheel: Up/Down • Enter to select • Esc to cancel"
            hint_surf = self._font_subtitle.render(hint, True, (160, 170, 180))
            self._display.blit(hint_surf, hint_surf.get_rect(center=(center_x, int(self._H * 0.92))))
            pygame.display.flip()

            # --- events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ids[sel] if ids else (_canon(default_id) or "Town01")

                # mapped joystick buttons
                if _is_mapped_press(event, map_up):
                    sel = (sel - 1) % len(names); continue
                if _is_mapped_press(event, map_down):
                    sel = (sel + 1) % len(names); continue
                if _is_mapped_press(event, map_enter):
                    return ids[sel]
                if _is_mapped_press(event, map_escape):
                    return _canon(default_id) or ids[sel]

                # keyboard fallback
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        sel = (sel - 1) % len(names)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        sel = (sel + 1) % len(names)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return ids[sel]
                    elif event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                        return _canon(default_id) or ids[sel]

                # D-pad hat
                if event.type == pygame.JOYHATMOTION and hat_cooldown == 0:
                    _, dy = event.value
                    if dy == 1:
                        sel = (sel - 1) % len(names); hat_cooldown = 5
                    elif dy == -1:
                        sel = (sel + 1) % len(names); hat_cooldown = 5

    def show_title_screen(self):
        """
        Title screen: welcome -> map Enter -> map Escape -> select vehicle -> continue.
        Returns (persistent_keys, chosen_vehicle_id).
        """
        # --- State & outputs ---
        state = "WELCOME"  # WELCOME, MAP_ENTER, MAP_ESCAPE, SELECT_VEHICLE, DONE, CONTINUE
        persistent_keys = {}  # must contain "Enter" and "Escape" -> {"joy_id":..., "button_id":...}
        chosen_vehicle_id = None
        carla_blueprint = None

        # --- Assets / fonts (unchanged from yours) ---
        logo_img = None
        try:
            logo_surface = pygame.image.load("./images/Logo_product.png").convert_alpha()
            original_size = logo_surface.get_size()
            scaled_size = (int(original_size[0] * 0.3), int(original_size[1] * 0.3))
            logo_img = pygame.transform.smoothscale(logo_surface, scaled_size)
        except pygame.error as e:
            logging.warning(f"Could not load logo image for title screen: {e}")

        # Colors & layout
        top_color, bottom_color = (44, 62, 80), (27, 38, 49)
        title_color, subtitle_color, prompt_color, complete_color = (
            (169, 204, 227), (189, 195, 199), (169, 204, 227), (0, 186, 6)
        )
        main_screen_offset_x = self._display.get_width() // 4
        single_screen_width = self._display.get_width() // 4
        center_x = main_screen_offset_x + (single_screen_width / 2)
        self.center_x = center_x

        title_surf = self._font_title.render("Safety Simulator", True, title_color)
        subtitle_surf = self._font_subtitle.render("", True, subtitle_color)
        author_surf = self._font_credits.render("Author: Arjun Joshi", True, (150, 150, 150))

        prompt_surf_welcome = self._font_prompt.render("Welcome to QRyde Sim", True, prompt_color)
        prompt_surf_sub     = self._font_prompt.render("Press any key/button • ESC to exit", True, prompt_color)
        prompt_surf_enter   = self._font_prompt.render("Press a WHEEL Button to Map [ENTER]", True, prompt_color)
        prompt_surf_up   = self._font_prompt.render("Press a WHEEL Button to Map [UP]", True, prompt_color)
        prompt_surf_down   = self._font_prompt.render("Press a WHEEL Button to Map [DOWN]", True, prompt_color)
        prompt_surf_escape  = self._font_prompt.render("Press a WHEEL Button to Map [ESCAPE]", True, prompt_color)
        prompt_surf_select  = self._font_prompt.render("Select a Vehicle from the Menu…", True, prompt_color)
        prompt_surf_map = self._font_prompt.render("Please Select a Map...",True, prompt_color)
        prompt_surf_done    = self._font_prompt.render("Mapped! Press any key/button to continue", True, complete_color)

        # Init joysticks
        pygame.joystick.init()
        logging.info(f"Welcome Screen. Num Joysticks: {pygame.joystick.get_count()}")
        for i in range(pygame.joystick.get_count()):
            js = pygame.joystick.Joystick(i); js.init()
            logging.info(f"Joystick {i} -> {js.get_name()} (id {js.get_instance_id()})")

        clock = pygame.time.Clock()

        while state != "CONTINUE":
            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and (event.key== pygame.K_c and (pygame.key.get_mods() & pygame.KMOD_CTRL)):
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                if state == "WELCOME":
                    if event.type in (pygame.KEYDOWN, pygame.JOYBUTTONDOWN, pygame.MOUSEBUTTONDOWN):
                        state = "MAP_ENTER"

                elif state == "MAP_ENTER":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["Enter"] = {"joy_id": event.instance_id, "button_id": event.button}
                        logging.info(f"Mapped [Enter] to joystick {event.instance_id}, button {event.button}")
                        state = "MAP_UP"
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        persistent_keys["Enter"] = {"joy_id": None, "button_id": None}
                        logging.info("Mapped [Enter] to keyboard Return")
                        state = "MAP_UP"
                elif state == "MAP_UP":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["Up"] = {"joy_id": event.instance_id, "button_id": event.button}
                        logging.info(f"Mapped [Up] to joystick {event.instance_id}, button {event.button}")
                        state = "MAP_DOWN"
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        persistent_keys["Up"] = {"joy_id": None, "button_id": None}  # keyboard fallback
                        logging.info("Mapped [Up] to keyboard Return")
                        state = "MAP_DOWN"

                elif state == "MAP_DOWN":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["Down"] = {"joy_id": event.instance_id, "button_id": event.button}
                        logging.info(f"Mapped [Down] to joystick {event.instance_id}, button {event.button}")
                        state = "MAP_ESCAPE"
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        persistent_keys["Down"] = {"joy_id": None, "button_id": None}  # keyboard fallback
                        logging.info("Mapped [Down] to keyboard Return")
                        state = "MAP_ESCAPE"
                elif state == "MAP_ESCAPE":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["Escape"] = {"joy_id": event.instance_id, "button_id": event.button}
                        logging.info(f"Mapped [Escape] to joystick {event.instance_id}, button {event.button}")
                        state = "SELECT_VEHICLE"
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        persistent_keys["Escape"] = {"joy_id": None, "button_id": None}
                        logging.info("Mapped [Escape] to keyboard Return")
                        state = "SELECT_VEHICLE"
                elif state == "DONE":
                    if event.type in (pygame.KEYDOWN, pygame.JOYBUTTONDOWN, pygame.MOUSEBUTTONDOWN):
                        state = "CONTINUE"

            # Trigger the selector exactly once when entering SELECT_VEHICLE
            if state == "SELECT_VEHICLE":
                chosen_vehicle_id, carla_blueprint = self.select_vehicle_config(persistent_keys,center_x, logo_img, default_id="ford_e450_super_duty")
                logging.info(f"Vehicle config selected: {chosen_vehicle_id}")
                state = "SELECT_MAP"
                # skip drawing the title frame this iteration; next frame will draw DONE prompt
                continue
            if state == "SELECT_MAP":
                chosen_map_id = self.select_map(persistent_keys,center_x, logo_img, default_id ="Town10HD")
                logging.info(f"Map Chosen: {chosen_map_id}")
                state = "DONE"
                continue
            
            # --- Draw (every frame, not only on events) ---
            # Gradient bg
            H = self._display.get_height(); W = self._display.get_width()
            for y in range(H):
                r = top_color[0] + (bottom_color[0] - top_color[0]) * y // H
                g = top_color[1] + (bottom_color[1] - top_color[1]) * y // H
                b = top_color[2] + (bottom_color[2] - top_color[2]) * y // H
                pygame.draw.line(self._display, (r, g, b), (0, y), (W, y))

            # Static
            if logo_img:
                self._display.blit(logo_img, logo_img.get_rect(center=(center_x, H * 0.30)))
            title_rect = title_surf.get_rect(center=(center_x, H * 0.55))
            subtitle_rect = subtitle_surf.get_rect(center=(center_x, title_rect.bottom))
            author_rect = author_surf.get_rect(center=(center_x, subtitle_rect.bottom))
            self._display.blit(title_surf, title_rect)
            self._display.blit(subtitle_surf, subtitle_rect)
            self._display.blit(author_surf, author_rect)

            # Dynamic prompt (blink)
            current = None
            if state == "WELCOME":       current = prompt_surf_welcome
            elif state == "MAP_ENTER":   current = prompt_surf_enter
            elif state == "MAP_UP": current = prompt_surf_up
            elif state == "MAP_DOWN": current = prompt_surf_down
            elif state == "MAP_ESCAPE":  current = prompt_surf_escape
            elif state == "SELECT_VEHICLE": current = prompt_surf_select
            elif state == "DONE":        current = prompt_surf_done

            if current and (pygame.time.get_ticks() // 500) % 2 == 0:
                self._display.blit(current, current.get_rect(center=(center_x, H * 0.77)))
                if state == "WELCOME":
                    self._display.blit(prompt_surf_sub, prompt_surf_sub.get_rect(center=(center_x, H * 0.82)))

            pygame.display.flip()
            clock.tick(60)

        if not chosen_vehicle_id:
            chosen_vehicle_id = "ford_e450_super_duty"
        if not chosen_map_id:
            chosen_map_id = "Town10HD_opt"
        logging.info("Title mapping & choice customization complete.")
        return persistent_keys, chosen_vehicle_id, carla_blueprint, chosen_map_id