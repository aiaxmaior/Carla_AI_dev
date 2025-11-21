import pygame
import sys
import os
import logging

import json
from VehicleLibrary import VehicleLibrary
#To be finished.
from ScenarioLibrary import ScenarioLibrary

# from VehicleLibrary import MapLibrary
# --- Dropdown widgets and consolidated selector ---
from typing import List, Tuple, Optional, Dict

# from pathlib import Path
from FontIconLibrary import IconLibrary, FontLibrary
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(filename)s:%(lineno)d - %(message)s')

ilib = IconLibrary()
flib = FontLibrary()


JOYSTICK_EVENTS = (pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION, pygame.JOYAXISMOTION)
ANY_BUTTON = (pygame.KEYDOWN, pygame.JOYBUTTONDOWN, pygame.MOUSEBUTTONDOWN)

def _check_quit(event, persistent_keys):
    """
    Hard-quit handler. Returns True iff it exits (though sys.exit() will terminate anyway).
    ESC, Ctrl+Q/C on keyboard, or mapped joystick 'ESCAPE'.
    """
    # 1) Window close
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
        return True  # unreachable but explicit

    # 2) Keyboard
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
            return True
        mods = pygame.key.get_mods()
        if (mods & pygame.KMOD_CTRL) and event.key in (pygame.K_q, pygame.K_c):
            pygame.quit()
            sys.exit()
            return True

    # 3) Joystick (button / hat / axis) mapped to 'ESCAPE'
    if event.type in (pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION, pygame.JOYAXISMOTION):
        esc_map = (persistent_keys or {}).get("ESCAPE")
        if esc_map and _is_mapped_event(event, esc_map):
            pygame.quit()
            sys.exit()
            return True

    return False
class _Dropdown:
    def __init__(
        self,
        title: str,
        options: List[Tuple[str, str]],
        index: int = 0,
        enabled: bool = True,
    ):
        self.title, self.options, self.index, self.enabled = (
            title,
            options[:],
            index,
            enabled,
        )
        if not self.options:
            self.options = [("None", "none")]
        self.index = max(0, min(self.index, len(self.options) - 1))

    @property
    def value(self):
        return self.options[self.index][1]

    @property
    def label(self):
        return self.options[self.index][0]

    def change(self, delta: int):
        if self.enabled and self.options:
            self.index = (self.index + delta) % len(self.options)

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)

    def draw(self, surface, x, y, w, focused, font, sub):
        pad, h = 10, 64
        bg = (26, 26, 26) if self.enabled else (16, 16, 16)
        bor = (255, 210, 70) if (focused and self.enabled) else (70, 70, 70)
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=12)
        pygame.draw.rect(surface, bor, (x, y, w, h), width=2, border_radius=12)
        surface.blit(
            sub.render(
                self.title, True, (200, 200, 200) if self.enabled else (120, 120, 120)
            ),
            (x + pad, y + 8),
        )
        surface.blit(
            font.render(
                self.label, True, (235, 235, 235) if self.enabled else (160, 160, 160)
            ),
            (x + pad, y + 28),
        )


def _list_vehicle_options(default_id: Optional[str] = None) -> Tuple[List[Tuple[str, str]], int, Dict[str, str]]:
    lib = VehicleLibrary()
    items = lib.list_display_items()  # [{'id','name','carla_blueprint', ...}]

    if not items:
        # Fallback: single option with a sensible blueprint
        opts = [("Ford E450 Paratransit", "mercedes_sprinter")]
        idx = 0
        bp_map = {"ford_e450_super_duty": "vehicle.ford.ambulance"}
        return opts, idx, bp_map

    opts = [(it["name"], it["id"]) for it in items]
    bp_map = {it["id"]: it.get("carla_blueprint") for it in items if it.get("id")}

    idx = 0
    if default_id:
        try:
            default_canon_id = lib.resolve_to_id(default_id)
            for i, (_, vid) in enumerate(opts):
                if vid == default_canon_id:
                    idx = i
                    break
        except Exception:
            pass

    return opts, idx, bp_map
"""
def _list_scenario_options(default_id: Optional[str] = None) -> Tuple[List[Tuple[str, str]], int]:

    #Uses VehicleLibrary to get a list of vehicles for the dropdown.
    #Returns a list of (label, value) tuples and the default index.

    lib = ScenarioLibrary()
    items = lib.list_display_items() # Gets [{'id': ..., 'name': ..., 'carla_blueprint':...}]

    if not items:
        # Fallback if no JSON configs are found
        return [("Mercedes Sprinter", "vehicle.mercedes.sprinter")], 0

    # Convert the list of dicts into the (label, value) format for the dropdown
    # Here, we use the vehicle's friendly name for the label and its ID for the value.
    opts = [(item["name"], item["id"]) for item in items]

    # Find the index of the default vehicle
    idx = 0
    if default_id:
        try:
            # Resolve any aliases (e.g., "sprinter") to the main ID from the JSON
            default_canon_id = lib.resolve_to_id(default_id)
            # Find the index of that ID in our options list
            for i, (_, vid) in enumerate(opts):
                if vid == default_canon_id:
                    idx = i
                    break
        except Exception:
            # If lookup fails, idx remains 0
            pass

    return opts, idx
"""

#### PLACEHOLDER UNTIL SCENARIO LIBRARY IS FINISHED
def _list_scenario_options(
    default_sid: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], int]:
    scenarios=ScenarioLibrary()
    opts,idx=scenarios.get_dropdown_options()
    return opts,idx
#### END PLACEHOLDER

def _list_map_options(client=None, prelisted=None, default_map=None):
    try:
        maps = prelisted or (client.get_available_maps() if client else [])
    except Exception:
        maps = []
    if not maps:
        maps = ["Town01", "Town02", "Town03", "Town10HD"]

    # normalize to (label, value)
    opts = [(m.split("/")[-1], m) for m in maps]

    idx = 0
    if default_map:
        for i, (_, mid) in enumerate(opts):
            if mid == default_map:
                idx = i
                break
    return opts, idx


def _is_mapped_event(event, mapping):
    """
    Helper function to check if a joystick event matches a specific mapping rule.
    """
    if not mapping or mapping.get("joy_idx") != getattr(event, "instance_id", -1):
        return False

    event_type = mapping.get("type", "button")

    if event_type == "button" and event.type == pygame.JOYBUTTONDOWN:
        return mapping.get("id") == event.button

    return False


def _nav_from_event(event, persistent_keys):
    """
    Handles all keyboard and joystick navigation for UI menus.
    Returns: "UP", "DOWN", "LEFT", "RIGHT", "ENTER", "ESCAPE", "EXIT", or None.
    """
    # --- Keyboard Input ---
    _check_quit(event,persistent_keys)

    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_UP, pygame.K_w):
            return "UP"
        if event.key in (pygame.K_DOWN, pygame.K_s):
            return "DOWN"
        if event.key in (pygame.K_LEFT, pygame.K_a):
            return "LEFT"
        if event.key in (pygame.K_RIGHT, pygame.K_d):
            return "RIGHT"
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "ENTER"
        if event.key == pygame.K_ESCAPE:
            return "ESCAPE"
        # Special exit hotkey
        if pygame.key.get_mods() & pygame.KMOD_CTRL and event.key == pygame.K_c:
            return "EXIT"

    # --- Mapped Joystick Input ---
    joystick_events = (pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION, pygame.JOYAXISMOTION)
    if event.type in joystick_events:
        # Check the event against each of our persistent key mappings
        for action in ["UP", "DOWN", "LEFT", "RIGHT", "ENTER", "ESCAPE"]:
            if _is_mapped_event(event, persistent_keys.get(action)):
                return action.upper()  # Return "UP", "DOWN", etc.

    return None


def consolidated_select_screen(
    display,
    args,
    splash_img,
    logo_img,
    persistent_keys,
    panel_x0,
    panel_w,
    scale_factor,
    client=None,
    anchor_index=None,
):
    _selections = {}
    _display = display
    _args = args
    client = client
    _H = _display.get_height()  # total window height
    _W = _display.get_width()  # total window width
    _splash_img = splash_img
    _logo_img = logo_img
    sizes = pygame.display.get_desktop_sizes() or [(_W, _H)]
    _num_panels = max(1, len(sizes))
    # assume all panels same height; take width from first (good enough for uniform setup)
    _panel_w, _panel_h = sizes[0][0], sizes[0][1]
    _overlay = pygame.Surface((_panel_w, _panel_h), pygame.SRCALPHA)

    # choose which monitor hosts the title UI
    # default to 0 (leftmost) when single-screen or when args.layout_mode == "single"
    _single_layout = (getattr(_args, "layout_mode", None) == "single") or (
        _num_panels == 1
    )
    _default_idx = 0 if _single_layout else 1

    # This function now receives all the info it needs. No need to recalculate.
    _display = display
    _H = _display.get_height()

    # Center on full window width (centers on ultrawide when using --display 1)
    center_x = _W // 2

    # All the code that was here trying to detect screens, calculate panel_w,
    # cli_idx, panel_x0, center_x, scale_factor, and load fonts
    # should be DELETED.

    # ----- fonts & UI text -----
    # The fonts are now passed in directly, so this line can be removed too:
    scale_factor = _H / 1080.0
    panel_fonts = flib.get_loaded_fonts(
        font="tt-supermolot-neue-trl.bd-it", type="mapping_screen", scale=scale_factor
    )

    prompt = panel_fonts["title"].render("Simulator Setup", True, (240, 240, 240))
    hint = panel_fonts["sub"].render(
        "UP/DOWN: move • LEFT/RIGHT: change • ENTER: confirm • Esc: back",
        True,
        (180, 180, 180),
    )
    joy_enter = getattr(args, "persistent_keys", {}).get("ENTER")

    def _is_mapped_press(ev, mapping):
        return (
            ev is not None and
            ev.type == pygame.JOYBUTTONDOWN and
            mapping and
            (mapping.get("joy_idx") == getattr(ev, "instance_id", getattr(ev, "joy", None))) and
            (mapping.get("id") == getattr(ev, "button", None))
        )
    def _scale_cover(img, dst_w, dst_h):
        iw, ih = img.get_size()
        sx, sy = dst_w / iw, dst_h / ih
        scale = max(sx, sy)  # cover (crop as needed), preserves aspect
        sw, sh = int(iw * scale), int(ih * scale)
        scaled = pygame.transform.smoothscale(img, (sw, sh))
        # center inside the dst panel
        surf = pygame.Surface((dst_w, dst_h)).convert()
        surf.blit(scaled, ((dst_w - sw) // 2, (dst_h - sh) // 2))
        return surf

    try:
        base = pygame.image.load("./images/welcomescreen_bus.png").convert_alpha()
        _splash_img = _scale_cover(base, _panel_w, _panel_h)
    except pygame.error as e:
        logging.warning(f"Could not load splash image: {e}")
        _splash_img = None
    # ----- fonts & UI text -----
    prompt = panel_fonts["main"].render("Simulator Setup", True, (240, 240, 240))
    hint = panel_fonts["sub"].render(
        "UP/DOWN: move • LEFT/RIGHT: change • ENTER: confirm • Esc: back",
        True,
        (180, 180, 180),
    )

    # ----- data sources -----
    sc_lib = ScenarioLibrary("./configs/scenarios")
    veh_opts, veh_idx, bp_map = _list_vehicle_options(getattr(args, "vehicle_id", None))
    scn_opts, scn_idx = sc_lib.get_dropdown_options(getattr(args, "scenario_id", "open_world"))
    map_opts, map_idx = _list_map_options(
        client=client, default_map=getattr(args, "map_id", None)
    )

    # Define all layout values at a base 1080p height
    # and then scale them by the current scale_factor.

    base_font_h = panel_fonts["sub"].get_height()
    DROPDOWN_BOX_HEIGHT = 144
    padding = int(base_font_h)  # e.g., 75% of a line of text

    # Calculate the scaled gap based on the base values and scale_factor
    row_gap = (DROPDOWN_BOX_HEIGHT + padding) * scale_factor

    # Starting Y position - shifted down 10% as requested
    y0 = _panel_h * 0.30

    # Scaled offsets for the text below the dropdowns
    help_text_offset = int(10 * scale_factor)
    hint_text_offset = int(80 * scale_factor)

    # The width and X position of the UI panel - centered on full window
    w = int(_panel_w * 0.8)
    x = center_x - (w // 2)  # Center dropdowns on full window center
    dd_vehicle = _Dropdown("Vehicle Selection", veh_opts, veh_idx, enabled=True)
    dd_scn = _Dropdown("Scenario Selector", scn_opts, scn_idx, enabled=True)
    dd_map = _Dropdown(
        "Map (Open World only)",
        map_opts,
        map_idx,
        enabled=(scn_opts[scn_idx][1] == "open_world"),
    )

    dropdowns = [dd_vehicle, dd_scn, dd_map]
    focus = 0
    clock = pygame.time.Clock()

    while True:
#        ilib.ilog(
#            "info",
#            f"DROPDOWN PANEL. panel_x0={panel_x0},_panel_h = {_panel_h}, y0={y0}, x = {x}",
#            "sysUI",
#            "dis",
#            2,
#       )
        center_x = center_x  # anchor all UI to screen #2 center
        display.fill((10, 10, 10))
        # Static Splash
        if _splash_img:
            _display.blit(_splash_img, (panel_x0, 0))
            _overlay.fill((0, 0, 0, 70))
            _display.blit(_overlay, (panel_x0, 0))
        else:
            _display.fill((24, 28, 34))
        # Static Logo
        if _logo_img:
            _display.blit(
                _logo_img, _logo_img.get_rect(center=(center_x, _panel_h * 0.15))
            )
        #            _overlay.fill((0, 0, 0, 70))  # 70–110 alpha looks nice

        # title centered on full window, positioned above dropdowns
        title_y = int(_panel_h * 0.22)  # Shifted down slightly
        display.blit(prompt, (center_x - prompt.get_width() // 2, title_y))

        # enable/disable map by scenario state
        dd_map.set_enabled(dd_scn.value == "open_world")

        # draw dropdowns using the new scaled row_gap
        for i, dd in enumerate(dropdowns):
            dd.draw(
                display,
                x,
                y0 + i * row_gap,
                w,
                focused=(i == focus),
                font=panel_fonts["sub"],
                sub=panel_fonts["sub_value"],
            )

        # scenario help line — also centered in the same panel
        scn_txt = {
            "collisions_s1": "Follow → surprise lead brake. Avoid collision.",
            "lane_mgmt_s1": "Hold center; one signaled lane change.",
            "driving_behavior_s1": "Smooth accel/brake; gentle turn.",
            "open_world": "Free roam with traffic.",
        }[dd_scn.value]

        sub_surf = panel_fonts["sub"].render(scn_txt, True, (200, 200, 200))
        # Position the help text relative to the last dropdown
        display.blit(
            sub_surf,
            (
                center_x - sub_surf.get_width() // 2,
                y0 + row_gap * len(dropdowns) + help_text_offset,
            ),
        )

        display.blit(
            hint,
            (
                center_x - hint.get_width() // 2,
                y0 + row_gap * len(dropdowns) + hint_text_offset,
            ),
        )

        pygame.display.flip()
        clock.tick(60)

        # Inside the consolidated_select_screen loop...
        for event in pygame.event.get():
            nav = _nav_from_event(event, persistent_keys)

            if nav == "UP":
                focus = (focus - 1) % len(dropdowns)
            elif nav == "DOWN":
                focus = (focus + 1) % len(dropdowns)
            elif nav == "LEFT":
                dropdowns[focus].change(-1)
            elif nav == "RIGHT":
                dropdowns[focus].change(1)
            elif nav in ("ESCAPE", "EXIT"):
                # This is the exit/cancel action
                # (You will need to break the loop and return default values here)
                if nav == "EXIT":
                    pygame.quit()
                    sys.exit()
                pass
            elif nav == "ENTER":
                args.vehicle_id  = dd_vehicle.value
                args.scenario_id = dd_scn.value
                args.map_id      = dd_map.value if dd_map.enabled else None

                chosen_vid = dd_vehicle.value
                bp = bp_map.get(chosen_vid)

                # Fallbacks: if your IDs are already full CARLA bps (start with "vehicle."),
                # use them directly; otherwise final fallback to a safe default.
                if not bp:
                    bp = chosen_vid if str(chosen_vid).startswith("vehicle.") else "vehicle.tesla.model3"

                setattr(args, "carla_blueprint", bp)
                return args
            elif _is_mapped_press(event, joy_enter):
                args.vehicle_id  = dd_vehicle.value
                args.scenario_id = dd_scn.value
                args.map_id      = dd_map.value if dd_map.enabled else None

                chosen_vid = dd_vehicle.value
                bp = bp_map.get(chosen_vid)

                # Fallbacks: if your IDs are already full CARLA bps (start with "vehicle."),
                # use them directly; otherwise final fallback to a safe default.
                if not bp:
                    bp = chosen_vid if str(chosen_vid).startswith("vehicle.") else "vehicle.tesla.model3"

                setattr(args, "carla_blueprint", bp)
                return args

class TitleScreen(object):
    def __init__(self, display, client, args):
        # self._persistent_keys_path= os.path.join(os.path.dirname(__file__), "configs", "joystick_mappings","welcome_mappings.json")
        self._persistent_keys_path = "./configs/joystick_mappings/welcome_mappings.json"
        self._default_id = "ford_e450_super_duty"

        # TitleScreen.__init__(...)
        self._display = display
        self._client = client
        self._H = self._display.get_height()  # total window height
        self._W = self._display.get_width()  # total window width
        self._args = args
        self._scale_factor = self._H / args.height

        ### FONT CLASS VARIABLES
        self.fonts_panel = flib.get_loaded_fonts(
            font="tt-supermolot-neue-trl.bd-it",
            type="select_screen",
            scale=self._scale_factor,
        )
        self._font_title = self.fonts_panel["title"]
        self._font_subtitle = self.fonts_panel["subtitle"]
        self._font_credits = self.fonts_panel["credits"]
        self._font_prompt = self.fonts_panel["prompt"]
        ##--------- RENDER CLASS VARIABLES [BEGIN]

        sizes = [(self._W, self._H)]
        self._num_panels = max(
            1, sizes[0][0] / self._args.width
        )  # estimate from window width
        # assume all panels same height; take width from first (good enough for uniform setup)
        total_w, self._panel_h = sizes[0][0], sizes[0][1]
        self._panel_w = total_w // self._num_panels
        ilib.ilog("warning", f"self.panel_w: {self._panel_w}", "alerts", "wn", 5)

        self._overlay = pygame.Surface((self._panel_w, self._panel_h), pygame.SRCALPHA)

        # choose which monitor hosts the title UI
        # default to 0 (leftmost) when single-screen or when args.layout_mode == "single"
        _single_layout = (getattr(self._args, "layout_mode", None) == "single") or (
            self._num_panels == 1
        )
        _default_idx = 0 if _single_layout else 1

        cli_idx = getattr(self._args, "title_screen_index", None)
        if cli_idx is None:
            self._title_screen_index = _default_idx
        else:
            # clamp to valid range
            try:
                self._title_screen_index = max(
                    0, min(int(cli_idx), self._num_panels - 1)
                )
            except Exception:
                self._title_screen_index = _default_idx
        self._panel_x0 = self._panel_w * self._title_screen_index
        ilib.ilog(f"title_screen_index: {self._title_screen_index}", "sysUI", "dis", 10)
        # Center on full window width (centers on ultrawide when using --display 1)
        self.center_x = self._W // 2

        def _scale_cover(img, dst_w, dst_h):
            iw, ih = img.get_size()
            sx, sy = dst_w / iw, dst_h / ih
            scale = max(sx, sy)  # cover (crop as needed), preserves aspect
            sw, sh = int(iw * scale), int(ih * scale)
            scaled = pygame.transform.smoothscale(img, (sw, sh))
            # center inside the dst panel
            surf = pygame.Surface((dst_w, dst_h)).convert()
            surf.blit(scaled, ((dst_w - sw) // 2, (dst_h - sh) // 2))
            return surf

        try:
            ilib.ilog(
                f"scale function defined. DROPDOWN PANEL. panel_x0={self._panel_x0}",
                "sysUI",
                "dis",
                2,
            )
            base = pygame.image.load("./images/welcomescreen_bus.png").convert_alpha()
            self._splash_img = _scale_cover(base, self._panel_w, self._panel_h)
        except pygame.error as e:
            logging.warning(f"Could not load splash image: {e}")
            self._splash_img = None

        try:
            logo_surface = pygame.image.load(
                "./images/logo_duhd.png"
            ).convert_alpha()
            # Logo is 3840x1080 (ultrawide) - scale to fit display width
            w, h = logo_surface.get_size()
            # Scale to fit 80% of display width while maintaining aspect ratio
            target_w = int(self._W * 0.8)
            scale = target_w / w
            self._logo_img = pygame.transform.smoothscale(
                logo_surface, (int(w * scale), int(h * scale))
            )
        except pygame.error as e:
            logging.warning(f"Could not load logo image for title screen: {e}")
            self._logo_img = None
            self._font_prompt = pygame.font.Font(None, 60)

        ##--------- RENDER CLASS VARIABLES [END]

    def _load(self):
        path = self._persistent_keys_path
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    mappings = json.load(f)
                if isinstance(mappings, dict):
                    return mappings
                else:
                    logging.warning(f"Invalid format in {path}. Expected a dictionary.")
            except json.JSONDecodeError as e:
                logging.warning(f"Error reading {path}: {e}. persistent_keys required.")
            except Exception as e:
                logging.warning(f"Unexpected error loading {path}: {e}")
        return None

    def _save_mappings(self, persistent_keys):
        """Saves the current joystick mappings to a JSON file."""
        path = self._persistent_keys_path
        try:
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                json.dump(persistent_keys, f, indent=4)
            logging.info(f"Joystick mappings saved to {path}")
        except Exception as e:
            logging.error(f"Could not save mappings: {e}")


    # Clean UI element for lists
    def _draw_panel(self, x_center, top_y, width_ratio=0.36, height_ratio=0.60):
        w = int(self._W * width_ratio / self._num_panels)  # ratio of ONE panel
        h = int(self._H * height_ratio)
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (x_center, top_y + h // 2)

        # subtle gradient
        gx, gy, gw, gh = rect
        top = (54, 61, 66)
        bot = (132, 137, 156)
        for y in range(gh):
            t = y / max(1, gh - 1)
            r = int(top[0] + (bot[0] - top[0]) * t)
            g = int(top[1] + (bot[1] - top[1]) * t)
            b = int(top[2] + (bot[2] - top[2]) * t)
            pygame.draw.line(self._display, (r, g, b), (gx, gy + y), (gx + gw, gy + y))
        pygame.draw.rect(self._display, (90, 100, 110), rect, width=2, border_radius=12)
        return rect

    # -------------------------------------------------------------------#
    # -------################### MAIN FUNCTION ###################-------#
    # -------------------------------------------------------------------#



    def show_title_screen(self):
        """
        Title screen: welcome -> map ENTER -> map ESCAPE -> select vehicle -> continue.
        Returns (persistent_keys, chosen_vehicle_id).
        """
        # --- State & outputs ---
        state = (
            "WELCOME"  # WELCOME, MAP_ENTER, MAP_ESCAPE, SELECT_VEHICLE, DONE, CONTINUE
        )
        persistent_keys = {}
        existing_mappings = False
        try:
            persistent_keys = self._load()
            if (
                persistent_keys is None
                or "ENTER" not in persistent_keys
                or "ESCAPE" not in persistent_keys
            ):
                persistent_keys = {}
            else:
                ilib.ilog(
                    "info", f"Loaded persistent keys: {persistent_keys}", "alerts", "i"
                )
                existing_mappings = True
        except Exception as e:
            ilib.ilog("info", f"Error loading persistent keys: {e}", "alerts", "e")
            pass

        chosen_vehicle_id = None
        carla_blueprint = None

        # --- Assets / fonts (unchanged from yours) ---

        # optional: a subtle dark overlay so text pops
        self._overlay.fill((0, 0, 0, 70))  # 70–110 alpha looks nice

        # Colors & layout
        top_color, bottom_color = (44, 62, 80), (27, 38, 49)
        title_color, subtitle_color, prompt_color, complete_color = (
            (169, 204, 227),
            (189, 195, 199),
            (169, 204, 227),
            (0, 186, 6),
        )
        #        main_screen_offset_x = self._display.get_width() // 4
        #        single_screen_width = self._display.get_width() // 4
        #        center_x = main_screen_offset_x + (single_screen_width / 2)
        #        self.center_x = center_x
        center_x = self.center_x
        title_surf = self._font_title.render("Safety Simulator", True, title_color)
        subtitle_surf = self._font_subtitle.render("", True, subtitle_color)
        author_surf = self._font_credits.render(
            "Author: Arjun Joshi", True, (150, 150, 150)
        )

        prompt_surf_welcome = self._font_prompt.render(
            "Welcome to QRyde Sim", True, prompt_color
        )
        prompt_surf_sub = self._font_prompt.render(
            "Press any key/button • ESC to exit", True, prompt_color
        )
        prompt_surf_esc_skip = self._font_prompt.render(
            "Press ESC to Exit Simulation or S to skip", True, prompt_color
        )
        prompt_surf_enter = self._font_prompt.render(
            "Press to Map [ENTER]", True, prompt_color
        )
        prompt_surf_up = self._font_prompt.render(
            "Press a WHEEL Button to Map [UP]", True, prompt_color
        )
        prompt_surf_down = self._font_prompt.render(
            "Press a WHEEL Button to Map [DOWN]", True, prompt_color
        )
        prompt_surf_left = self._font_prompt.render(
            "Press a WHEEL Button to Map [LEFT]", True, prompt_color
        )
        prompt_surf_right = self._font_prompt.render(
            "Press a WHEEL Button to Map [RIGHT]", True, prompt_color
        )
        prompt_surf_escape = self._font_prompt.render(
            "Press a WHEEL Button to Map [ESCAPE]", True, prompt_color
        )
        prompt_surf_done = self._font_prompt.render(
            "Mapped! Press any key/button to continue", True, complete_color
        )

        # Init joysticks
        pygame.joystick.init()
        logging.info(f"Welcome Screen. Num Joysticks: {pygame.joystick.get_count()}")
        for i in range(pygame.joystick.get_count()):
            js = pygame.joystick.Joystick(i)
            js.init()
            logging.info(f"Joystick {i} -> {js.get_name()} (id {js.get_instance_id()})")

        clock = pygame.time.Clock()

        while state != "CONTINUE":
            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type in ANY_BUTTON or JOYSTICK_EVENTS:
                    if _check_quit(event,persistent_keys) and existing_mappings:
                        pygame.quit()
                        sys.exit()
                    else:
                        pass

                # WECOME SCREEN
                if state == "WELCOME":
                    if event.type == pygame.KEYDOWN and (
                        event.key == pygame.K_s
                        and (pygame.key.get_mods() & pygame.KMOD_CTRL)
                    ):
                        logging.info("⚠️Steering Wheel mapping skipped")
                        if existing_mappings:
                            state = "SELECT_STAGE"
                        else:
                            state = "MAP_ENTER"
                    elif event.type in ANY_BUTTON:
                        if existing_mappings:
                            state = "SELECT_STAGE"
                        else:
                            state = "MAP_ENTER"

                # PERFORM SELECTION STAGE
                elif state == "SELECT_STAGE":
                    self._args=consolidated_select_screen(
                        self._display,
                        self._args,
                        self._splash_img,
                        self._logo_img,
                        persistent_keys,
                        self._panel_x0,
                        self._panel_w,
                        self._scale_factor,
                        client=self._client,
                        anchor_index=None,
                    )
                    ilib.ilog("info", "Selections Complete.", "alerts", "s")
                    state = "DONE"
                # BEGIN BASIC MAPPING
                elif state == "MAP_ENTER":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["ENTER"] = {
                            "type":"button",
                            "joy_idx": event.instance_id,
                            "id": event.button,
                        }
                        logging.info(
                            f"Mapped [ENTER] to joystick {event.instance_id}, button {event.button}"
                        )
                        state = "MAP_UP"

                elif state == "MAP_UP":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["UP"] = {
                            "type":"button",
                            "joy_idx": event.instance_id,
                            "id": event.button,
                        }
                        logging.info(
                            f"Mapped [UP] to joystick {event.instance_id}, button {event.button}"
                        )
                        state = "MAP_DOWN"
                    pygame.time.wait(300)  # debounce

                # Map DOWN
                elif state == "MAP_DOWN":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["DOWN"] = {
                            "type":"button",
                            "joy_idx": event.instance_id,
                            "id": event.button,
                        }
                        logging.info(
                            f"Mapped [DOWN] to joystick {event.instance_id}, button {event.button}"
                        )
                        state = "MAP_LEFT"

                # Map LEFT
                elif state == "MAP_LEFT":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["LEFT"] = {
                            "type":"button",
                            "joy_idx": event.instance_id,
                            "id": event.button,
                        }
                        logging.info(
                            f"Mapped [LEFT] to joystick {event.instance_id}, button {event.button}"
                        )
                        state = "MAP_RIGHT"

                # Map RIGHT
                elif state == "MAP_RIGHT":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["RIGHT"] = {
                            "type":"button",
                            "joy_idx": event.instance_id,
                            "id": event.button,
                        }
                        logging.info(
                            f"Mapped [RIGHT] to joystick {event.instance_id}, button {event.button}"
                        )
                        state = "MAP_ESCAPE"
                # Map ESCAPE
                elif state == "MAP_ESCAPE":
                    if event.type == pygame.JOYBUTTONDOWN:
                        persistent_keys["ESCAPE"] = {
                            "type":"button",
                            "joy_idx": event.instance_id,
                            "id": event.button,
                        }
                        logging.info(
                            f"Mapped [ESCAPE] to joystick {event.instance_id}, button {event.button}"
                        )
                        try:
                            self._save_mappings(persistent_keys)
                            ilib.ilog("info", "Mappings saved.", "file_data", "s")
                        except Exception as e:
                            ilib.ilog("info",f"Error saving mappings: {e}", 'alerts',"e")
                        state = "SELECT_STAGE"

                    elif event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_RETURN,
                        pygame.K_KP_ENTER,
                    ):
                        persistent_keys["ESCAPE"] = {
                            "type":"button",
                            "joy_idx": None,
                            "id": None,
                        }  # keyboard fallback
                        logging.info("Mapped [ESCAPE] to keyboard Return")
                        state = "MAP_ESCAPE"

                        ### CHECK FOR EXISTING MAPPINGS FOR SAVING
                        if not existing_mappings:
                            try:
                                self._save_mappings(persistent_keys)
                                ilib.ilog("info", "Mappings saved.", "file_data", "s")
                                pass
                            except Exception as e:
                                ilib.ilog(
                                    "info",
                                    f"Error saving mappings: {e}",
                                    "alerts",
                                    "e",
                                )
                        else:
                            ilib.ilog(
                                "info",
                                "Existing mappings were loaded; not overwriting.",
                                "alerts",
                                "i",
                            )
                        state = "SELECT_STAGE"

                elif state == "DONE":
                    if event.type in (
                        pygame.KEYDOWN,
                        pygame.JOYBUTTONDOWN,
                        pygame.MOUSEBUTTONDOWN,
                    ):
                        state = "CONTINUE"

            # Trigger the selector exactly once when entering SELECT_VEHICLE
            if state == "SELECT_VEHICLE":
                chosen_vehicle_id, carla_blueprint = self.select_vehicle_config(
                    persistent_keys, center_x, default_id="ford_e450_super_duty"
                )
                logging.info(f"🚗 Vehicle config selected: {chosen_vehicle_id}")
                state = "SELECT_MAP"
                # skip drawing the title frame this iteration; next frame will draw DONE prompt
                continue
            if state == "SELECT_MAP":
                chosen_map_id = self.select_map(
                    persistent_keys, center_x, default_id="Town10HD"
                )
                logging.info(f"🗺️Map Chosen: {chosen_map_id}")
                state = "DONE"
                continue

            # --- Draw (every frame, not only on events) ---
            # Gradient bg
            H = self._display.get_height()
            W = self._display.get_width()
            for y in range(H):
                r = top_color[0] + (bottom_color[0] - top_color[0]) * y // H
                g = top_color[1] + (bottom_color[1] - top_color[1]) * y // H
                b = top_color[2] + (bottom_color[2] - top_color[2]) * y // H
                pygame.draw.line(self._display, (r, g, b), (0, y), (W, y))

            # Static Splash
            if self._splash_img:
                self._display.blit(self._splash_img, (self._panel_x0, 0))
                self._overlay.fill((0, 0, 0, 70))
                self._display.blit(self._overlay, (self._panel_x0, 0))
            else:
                self._display.fill((24, 28, 34))
            # Static Logo
            if self._logo_img:
                self._display.blit(
                    self._logo_img, self._logo_img.get_rect(center=(center_x, H * 0.15))
                )
            #            self._overlay.fill((0, 0, 0, 70))  # 70–110 alpha looks nice

            title_rect = title_surf.get_rect(center=(center_x, H * 0.33))
            subtitle_rect = subtitle_surf.get_rect(center=(center_x, title_rect.bottom))
            author_rect = author_surf.get_rect(center=(center_x, subtitle_rect.bottom))
            self._display.blit(title_surf, title_rect)
            self._display.blit(subtitle_surf, subtitle_rect)
            self._display.blit(author_surf, author_rect)

            # Dynamic prompt (blink)
            current = None
            if state == "WELCOME":
                current = prompt_surf_welcome
            elif state == "MAP_ENTER":
                current = prompt_surf_enter
            elif state == "MAP_UP":
                current = prompt_surf_up
            elif state == "MAP_DOWN":
                current = prompt_surf_down
            elif state == "MAP_LEFT":
                current = prompt_surf_left
            elif state == "MAP_RIGHT":
                current = prompt_surf_right
            elif state == "MAP_ESCAPE":
                current = prompt_surf_escape
            elif state == "DONE":
                current = prompt_surf_done

            if current and (pygame.time.get_ticks() // 750) % 2 == 0:
                self._display.blit(
                    current, current.get_rect(center=(center_x, H * 0.77))
                )

                if state == "WELCOME":
                    self._display.blit(
                        prompt_surf_sub,
                        prompt_surf_sub.get_rect(center=(center_x, H * 0.82)),
                    )
                else:
                    self._display.blit(
                        prompt_surf_esc_skip,
                        prompt_surf_esc_skip.get_rect(center=(center_x, H * 0.82)),
                    )

            pygame.display.flip()
            clock.tick(60)

        if not getattr(self._args,'scenario_id',None):
            self._args.scenario_id = 'open_world'
        carla_blueprint = getattr(self._args,'carla_blueprint',None)
        if not carla_blueprint:
            vid = getattr(self._args,"vehicle_id","")
            carla_blueprint = vid if str(vid).startswith('vehicle.') else "vehicle.ford.ambulance"

        return persistent_keys, self._args.vehicle_id, carla_blueprint, self._args.map_id
