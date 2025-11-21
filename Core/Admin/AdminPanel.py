"""
Administrator Panel for Q-DRIVE Alpha

A standalone GUI panel for simulator administrators to:
1. Configure vehicle hyperparameters (steering sensitivity, physics)
2. View real-time driver performance metrics
3. Optionally replace PreWelcomeSelect for initial setup
4. View/modify custom_assets vehicle blueprints

Usage:
    # Standalone mode
    python -m Core.Admin.AdminPanel --standalone

    # From Main.py with --admin-panel argument
    python Main.py --admin-panel

    # Replace PreWelcomeSelect
    python Main.py --admin-setup

Author: Claude Code
Date: 2024
"""

import pygame
import json
import os
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from enum import Enum

# Initialize pygame for standalone use
pygame.init()


class AdminPanelMode(Enum):
    STANDALONE = "standalone"      # Independent window on 3rd monitor
    EMBEDDED = "embedded"          # Embedded in simulation
    SETUP = "setup"                # Replace PreWelcomeSelect


@dataclass
class VehicleHyperparameters:
    """Hyperparameters affecting vehicle movement for administrator tuning."""
    # Steering
    steer_sensitivity: float = 1.0
    steer_deadzone: float = 0.05
    steer_linearity: float = 1.0
    max_steering_angle: float = 70.0

    # Throttle/Brake
    throttle_sensitivity: float = 1.0
    brake_sensitivity: float = 1.0
    pedal_deadzone: float = 0.05

    # Physics
    mass_multiplier: float = 1.0
    drag_coefficient: float = 0.3
    tire_friction: float = 3.0

    # Ackermann Steering
    ackermann_enabled: bool = True
    wheelbase: float = 2.87  # meters
    track_width: float = 1.58  # meters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steering": {
                "sensitivity": self.steer_sensitivity,
                "deadzone": self.steer_deadzone,
                "linearity": self.steer_linearity,
                "max_angle": self.max_steering_angle,
            },
            "pedals": {
                "throttle_sensitivity": self.throttle_sensitivity,
                "brake_sensitivity": self.brake_sensitivity,
                "deadzone": self.pedal_deadzone,
            },
            "physics": {
                "mass_multiplier": self.mass_multiplier,
                "drag_coefficient": self.drag_coefficient,
                "tire_friction": self.tire_friction,
            },
            "ackermann": {
                "enabled": self.ackermann_enabled,
                "wheelbase": self.wheelbase,
                "track_width": self.track_width,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleHyperparameters':
        return cls(
            steer_sensitivity=data.get("steering", {}).get("sensitivity", 1.0),
            steer_deadzone=data.get("steering", {}).get("deadzone", 0.05),
            steer_linearity=data.get("steering", {}).get("linearity", 1.0),
            max_steering_angle=data.get("steering", {}).get("max_angle", 70.0),
            throttle_sensitivity=data.get("pedals", {}).get("throttle_sensitivity", 1.0),
            brake_sensitivity=data.get("pedals", {}).get("brake_sensitivity", 1.0),
            pedal_deadzone=data.get("pedals", {}).get("deadzone", 0.05),
            mass_multiplier=data.get("physics", {}).get("mass_multiplier", 1.0),
            drag_coefficient=data.get("physics", {}).get("drag_coefficient", 0.3),
            tire_friction=data.get("physics", {}).get("tire_friction", 3.0),
            ackermann_enabled=data.get("ackermann", {}).get("enabled", True),
            wheelbase=data.get("ackermann", {}).get("wheelbase", 2.87),
            track_width=data.get("ackermann", {}).get("track_width", 1.58),
        )


@dataclass
class DriverPerformanceMetrics:
    """Real-time driver performance metrics for admin monitoring."""
    # MVD Scores
    overall_score: float = 100.0
    collision_avoidance: float = 100.0
    lane_discipline: float = 100.0
    driving_smoothness: float = 100.0

    # Predictive Indices
    ttc_s: float = 99.0
    tlc_s: float = 99.0
    p_collision: float = 0.0
    p_lane_violation: float = 0.0

    # Vehicle State
    speed_kmh: float = 0.0
    rpm: float = 0.0
    gear: str = "N"
    steering_angle: float = 0.0

    # Events
    total_collisions: int = 0
    total_lane_violations: int = 0
    total_harsh_events: int = 0

    # DMS (if available)
    attention_score: Optional[float] = None
    drowsiness_score: Optional[float] = None
    distraction_score: Optional[float] = None


class AdminPanel:
    """
    Administrator Panel GUI for Q-DRIVE Alpha.

    Can run as:
    - Standalone window on 3rd monitor (max 1024x768)
    - Embedded in simulation
    - Pre-simulation setup (replace PreWelcomeSelect)
    """

    CONFIG_FILE = "./configs/admin_hyperparameters.json"

    def __init__(self, mode: AdminPanelMode = AdminPanelMode.STANDALONE,
                 width: int = 1024, height: int = 768):
        self.mode = mode
        self.width = min(width, 1024)  # Max 1024 as specified
        self.height = height

        self.hyperparams = VehicleHyperparameters()
        self.metrics = DriverPerformanceMetrics()

        self.running = False
        self.display = None
        self.clock = None

        # UI State
        self.selected_tab = "hyperparams"  # hyperparams, metrics, setup
        self.selected_param_idx = 0
        self.editing_value = False
        self.edit_buffer = ""

        # Callbacks for live updates
        self.on_param_change: Optional[Callable[[VehicleHyperparameters], None]] = None

        # Load saved config
        self._load_config()

        # Colors
        self.colors = {
            "background": (30, 30, 40),
            "panel": (45, 45, 55),
            "header": (60, 60, 80),
            "text": (220, 220, 220),
            "highlight": (100, 150, 255),
            "warning": (255, 200, 50),
            "danger": (255, 80, 80),
            "success": (80, 255, 80),
        }

        # Fonts (lazy init)
        self._fonts = None

    @property
    def fonts(self):
        if self._fonts is None:
            self._fonts = {
                "title": pygame.font.Font(None, 36),
                "header": pygame.font.Font(None, 28),
                "body": pygame.font.Font(None, 22),
                "small": pygame.font.Font(None, 18),
            }
        return self._fonts

    def _load_config(self):
        """Load hyperparameters from config file."""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.hyperparams = VehicleHyperparameters.from_dict(data)
                logging.info(f"Loaded admin config from {self.CONFIG_FILE}")
            except Exception as e:
                logging.warning(f"Failed to load admin config: {e}")

    def _save_config(self):
        """Save hyperparameters to config file."""
        os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.hyperparams.to_dict(), f, indent=2)
            logging.info(f"Saved admin config to {self.CONFIG_FILE}")
        except Exception as e:
            logging.error(f"Failed to save admin config: {e}")

    def update_metrics(self, metrics: DriverPerformanceMetrics):
        """Update driver performance metrics (called from main simulation)."""
        self.metrics = metrics

    def update_metrics_from_dict(self, data: Dict[str, Any]):
        """Update metrics from a dictionary (convenience method)."""
        self.metrics.overall_score = data.get('overall_score', 100.0)
        self.metrics.collision_avoidance = data.get('collision_avoidance', 100.0)
        self.metrics.lane_discipline = data.get('lane_discipline', 100.0)
        self.metrics.driving_smoothness = data.get('driving_smoothness', 100.0)
        self.metrics.ttc_s = data.get('ttc_s', 99.0)
        self.metrics.tlc_s = data.get('tlc_s', 99.0)
        self.metrics.p_collision = data.get('p_collision', 0.0)
        self.metrics.p_lane_violation = data.get('p_lane_violation', 0.0)
        self.metrics.speed_kmh = data.get('speed_kmh', 0.0)
        self.metrics.rpm = data.get('rpm', 0.0)
        self.metrics.gear = data.get('gear', 'N')
        self.metrics.steering_angle = data.get('steering_angle', 0.0)
        self.metrics.total_collisions = data.get('total_collisions', 0)
        self.metrics.total_lane_violations = data.get('total_lane_violations', 0)
        self.metrics.total_harsh_events = data.get('total_harsh_events', 0)
        self.metrics.attention_score = data.get('attention_score')
        self.metrics.drowsiness_score = data.get('drowsiness_score')
        self.metrics.distraction_score = data.get('distraction_score')

    def get_hyperparams(self) -> VehicleHyperparameters:
        """Get current hyperparameters for use by simulation."""
        return self.hyperparams

    def _init_display(self):
        """Initialize pygame display for standalone mode."""
        if self.mode == AdminPanelMode.STANDALONE:
            # Try to position on 3rd monitor if available
            os.environ['SDL_VIDEO_WINDOW_POS'] = "2048,0"  # Approximate 3rd monitor
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Q-DRIVE Admin Panel")
        self.clock = pygame.time.Clock()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.editing_value:
                        self.editing_value = False
                        self.edit_buffer = ""
                    else:
                        self.running = False
                elif event.key == pygame.K_TAB:
                    # Switch tabs
                    tabs = ["hyperparams", "metrics", "setup"]
                    idx = tabs.index(self.selected_tab)
                    self.selected_tab = tabs[(idx + 1) % len(tabs)]
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self._save_config()
                elif event.key == pygame.K_UP:
                    self.selected_param_idx = max(0, self.selected_param_idx - 1)
                elif event.key == pygame.K_DOWN:
                    self.selected_param_idx += 1
                elif event.key == pygame.K_RETURN:
                    if self.editing_value:
                        self._apply_edit()
                    else:
                        self.editing_value = True
                elif self.editing_value:
                    if event.key == pygame.K_BACKSPACE:
                        self.edit_buffer = self.edit_buffer[:-1]
                    elif event.unicode.isprintable():
                        self.edit_buffer += event.unicode

    def _apply_edit(self):
        """Apply edited value to hyperparameters."""
        # Get param list and apply edit
        params = self._get_param_list()
        if 0 <= self.selected_param_idx < len(params):
            name, attr, _ = params[self.selected_param_idx]
            try:
                value = float(self.edit_buffer)
                setattr(self.hyperparams, attr, value)
                if self.on_param_change:
                    self.on_param_change(self.hyperparams)
            except ValueError:
                pass
        self.editing_value = False
        self.edit_buffer = ""

    def _get_param_list(self):
        """Get list of editable parameters."""
        return [
            ("Steer Sensitivity", "steer_sensitivity", self.hyperparams.steer_sensitivity),
            ("Steer Deadzone", "steer_deadzone", self.hyperparams.steer_deadzone),
            ("Steer Linearity", "steer_linearity", self.hyperparams.steer_linearity),
            ("Max Steering Angle", "max_steering_angle", self.hyperparams.max_steering_angle),
            ("Throttle Sensitivity", "throttle_sensitivity", self.hyperparams.throttle_sensitivity),
            ("Brake Sensitivity", "brake_sensitivity", self.hyperparams.brake_sensitivity),
            ("Pedal Deadzone", "pedal_deadzone", self.hyperparams.pedal_deadzone),
            ("Mass Multiplier", "mass_multiplier", self.hyperparams.mass_multiplier),
            ("Drag Coefficient", "drag_coefficient", self.hyperparams.drag_coefficient),
            ("Tire Friction", "tire_friction", self.hyperparams.tire_friction),
            ("Wheelbase (m)", "wheelbase", self.hyperparams.wheelbase),
            ("Track Width (m)", "track_width", self.hyperparams.track_width),
        ]

    def _render(self):
        """Render the admin panel."""
        if self.display is None:
            return

        self.display.fill(self.colors["background"])

        # Header
        header_rect = pygame.Rect(0, 0, self.width, 50)
        pygame.draw.rect(self.display, self.colors["header"], header_rect)
        title = self.fonts["title"].render("Q-DRIVE Administrator Panel", True, self.colors["text"])
        self.display.blit(title, (20, 12))

        # Tab buttons
        tabs = [("hyperparams", "Hyperparameters"), ("metrics", "Driver Metrics"), ("setup", "Setup")]
        tab_x = 20
        for tab_id, tab_name in tabs:
            color = self.colors["highlight"] if self.selected_tab == tab_id else self.colors["panel"]
            tab_surf = self.fonts["body"].render(tab_name, True, self.colors["text"])
            tab_rect = pygame.Rect(tab_x, 60, tab_surf.get_width() + 20, 30)
            pygame.draw.rect(self.display, color, tab_rect, border_radius=5)
            self.display.blit(tab_surf, (tab_x + 10, 65))
            tab_x += tab_surf.get_width() + 30

        # Content area
        content_y = 100
        if self.selected_tab == "hyperparams":
            self._render_hyperparams(content_y)
        elif self.selected_tab == "metrics":
            self._render_metrics(content_y)
        elif self.selected_tab == "setup":
            self._render_setup(content_y)

        # Footer
        footer_y = self.height - 30
        footer = self.fonts["small"].render("Ctrl+S: Save | Tab: Switch Tab | Esc: Exit | Up/Down: Navigate | Enter: Edit",
                                            True, (150, 150, 150))
        self.display.blit(footer, (20, footer_y))

        pygame.display.flip()

    def _render_hyperparams(self, start_y: int):
        """Render hyperparameters tab."""
        params = self._get_param_list()
        y = start_y

        for idx, (name, attr, value) in enumerate(params):
            selected = idx == self.selected_param_idx

            # Background for selected item
            if selected:
                bg_rect = pygame.Rect(10, y - 2, self.width - 20, 26)
                pygame.draw.rect(self.display, self.colors["highlight"], bg_rect, border_radius=3)

            # Name
            name_surf = self.fonts["body"].render(name, True, self.colors["text"])
            self.display.blit(name_surf, (20, y))

            # Value
            if selected and self.editing_value:
                value_text = self.edit_buffer + "_"
            else:
                value_text = f"{value:.3f}" if isinstance(value, float) else str(value)
            value_surf = self.fonts["body"].render(value_text, True, self.colors["text"])
            self.display.blit(value_surf, (self.width - value_surf.get_width() - 20, y))

            y += 28

    def _render_metrics(self, start_y: int):
        """Render driver performance metrics tab."""
        y = start_y

        # MVD Scores Section
        section_title = self.fonts["header"].render("MVD Scores", True, self.colors["highlight"])
        self.display.blit(section_title, (20, y))
        y += 35

        metrics_list = [
            ("Overall Score", self.metrics.overall_score, 100.0),
            ("Collision Avoidance", self.metrics.collision_avoidance, 100.0),
            ("Lane Discipline", self.metrics.lane_discipline, 100.0),
            ("Driving Smoothness", self.metrics.driving_smoothness, 100.0),
        ]

        for name, value, max_val in metrics_list:
            self._render_metric_bar(20, y, name, value, max_val)
            y += 30

        # Predictive Indices Section
        y += 20
        section_title = self.fonts["header"].render("Predictive Indices", True, self.colors["highlight"])
        self.display.blit(section_title, (20, y))
        y += 35

        indices = [
            ("TTC (s)", f"{self.metrics.ttc_s:.1f}"),
            ("TLC (s)", f"{self.metrics.tlc_s:.1f}"),
            ("P(Collision)", f"{self.metrics.p_collision:.2%}"),
            ("P(Lane Violation)", f"{self.metrics.p_lane_violation:.2%}"),
        ]

        for name, value_str in indices:
            text = self.fonts["body"].render(f"{name}: {value_str}", True, self.colors["text"])
            self.display.blit(text, (20, y))
            y += 25

        # Vehicle State Section
        y += 20
        section_title = self.fonts["header"].render("Vehicle State", True, self.colors["highlight"])
        self.display.blit(section_title, (20, y))
        y += 35

        vehicle = [
            f"Speed: {self.metrics.speed_kmh:.1f} km/h",
            f"RPM: {self.metrics.rpm:.0f}",
            f"Gear: {self.metrics.gear}",
            f"Steering: {self.metrics.steering_angle:.1f} deg",
        ]

        for text_str in vehicle:
            text = self.fonts["body"].render(text_str, True, self.colors["text"])
            self.display.blit(text, (20, y))
            y += 25

        # DMS Section (if available)
        if self.metrics.attention_score is not None:
            y += 20
            section_title = self.fonts["header"].render("Driver Monitoring (DMS)", True, self.colors["highlight"])
            self.display.blit(section_title, (20, y))
            y += 35

            dms_metrics = [
                ("Attention", self.metrics.attention_score or 0, 1.0),
                ("Drowsiness", self.metrics.drowsiness_score or 0, 1.0),
                ("Distraction", self.metrics.distraction_score or 0, 1.0),
            ]

            for name, value, max_val in dms_metrics:
                self._render_metric_bar(20, y, name, value * 100, max_val * 100)
                y += 30

    def _render_metric_bar(self, x: int, y: int, name: str, value: float, max_val: float):
        """Render a metric with progress bar."""
        bar_width = 200
        bar_height = 18

        # Name
        name_surf = self.fonts["body"].render(name, True, self.colors["text"])
        self.display.blit(name_surf, (x, y))

        # Bar background
        bar_x = x + 180
        pygame.draw.rect(self.display, self.colors["panel"], (bar_x, y, bar_width, bar_height), border_radius=3)

        # Bar fill
        fill_pct = max(0, min(1, value / max_val))
        fill_width = int(bar_width * fill_pct)
        if fill_pct > 0.7:
            fill_color = self.colors["success"]
        elif fill_pct > 0.4:
            fill_color = self.colors["warning"]
        else:
            fill_color = self.colors["danger"]
        pygame.draw.rect(self.display, fill_color, (bar_x, y, fill_width, bar_height), border_radius=3)

        # Value text
        value_surf = self.fonts["small"].render(f"{value:.1f}", True, self.colors["text"])
        self.display.blit(value_surf, (bar_x + bar_width + 10, y))

    def _render_setup(self, start_y: int):
        """Render setup tab (PreWelcomeSelect replacement)."""
        y = start_y

        title = self.fonts["header"].render("Simulation Setup", True, self.colors["highlight"])
        self.display.blit(title, (20, y))
        y += 40

        instructions = [
            "This panel can replace PreWelcomeSelect for initial setup.",
            "",
            "Available Options:",
            "  - Select vehicle from library",
            "  - Choose map/scenario",
            "  - Configure traffic density",
            "  - Set weather conditions",
            "  - Configure MVD penalty settings",
            "",
            "Use --admin-setup to launch directly to this panel.",
            "",
            "Note: Full setup functionality coming soon.",
        ]

        for line in instructions:
            text = self.fonts["body"].render(line, True, self.colors["text"])
            self.display.blit(text, (20, y))
            y += 24

    def run_standalone(self):
        """Run the admin panel in standalone mode (separate window)."""
        self._init_display()
        self.running = True

        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(30)

        pygame.quit()

    def run_threaded(self):
        """Run the admin panel in a separate thread (for use during simulation)."""
        thread = threading.Thread(target=self.run_standalone, daemon=True)
        thread.start()
        return thread

    def render_to_surface(self, surface: pygame.Surface, x: int, y: int):
        """Render the admin panel to an existing surface (embedded mode)."""
        # Create a subsurface and render to it
        sub = surface.subsurface(pygame.Rect(x, y, self.width, self.height))
        self.display = sub
        self._render()


def main():
    """Main entry point for standalone admin panel."""
    import argparse
    parser = argparse.ArgumentParser(description="Q-DRIVE Administrator Panel")
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode")
    parser.add_argument("--width", type=int, default=1024, help="Window width (max 1024)")
    parser.add_argument("--height", type=int, default=768, help="Window height")
    args = parser.parse_args()

    panel = AdminPanel(
        mode=AdminPanelMode.STANDALONE,
        width=args.width,
        height=args.height
    )
    panel.run_standalone()


if __name__ == "__main__":
    main()
