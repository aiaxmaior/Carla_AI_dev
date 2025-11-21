"""
Experimental Audio Engine for Q-DRIVE Alpha

A modular audio framework for introducing realistic driving sounds:
- Engine sound modulation based on RPM, acceleration
- Vehicle proximity sounds (other cars, pedestrians)
- Ambient environmental sounds
- Honking behaviors
- Wind/road noise

EXPERIMENTAL: This module is optional and activates only when --enable-audio is passed.

Usage:
    # From command line
    python Main.py --enable-audio

    # In code
    from Core.Audio import AudioEngine
    audio = AudioEngine()
    audio.start()
    ...
    audio.update(rpm=3500, speed_kmh=60, ...)
    ...
    audio.stop()

Author: Claude Code
Date: 2024
"""

import pygame
import numpy as np
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class SoundCategory(Enum):
    """Categories of sounds in the simulation."""
    ENGINE = "engine"
    EXHAUST = "exhaust"
    WIND = "wind"
    ROAD = "road"
    VEHICLE = "vehicle"       # Other vehicles
    PEDESTRIAN = "pedestrian"
    HORN = "horn"
    BLINKER = "blinker"
    COLLISION = "collision"
    AMBIENT = "ambient"


@dataclass
class SoundSource:
    """Represents a positioned sound source in 3D space."""
    category: SoundCategory
    position: Tuple[float, float, float] = (0, 0, 0)  # x, y, z in world coords
    velocity: Tuple[float, float, float] = (0, 0, 0)  # For Doppler effect
    volume: float = 1.0
    pitch: float = 1.0
    looping: bool = False
    active: bool = True
    sound_file: Optional[str] = None
    channel: Optional[pygame.mixer.Channel] = None


@dataclass
class EngineState:
    """Current engine state for sound modulation."""
    rpm: float = 0.0
    throttle: float = 0.0
    gear: int = 0
    speed_kmh: float = 0.0
    acceleration: float = 0.0
    load: float = 0.0  # Engine load (0-1)


@dataclass
class AudioConfig:
    """Configuration for audio engine."""
    # Master controls
    master_volume: float = 0.8
    enabled: bool = False

    # Category volumes
    engine_volume: float = 0.7
    exhaust_volume: float = 0.5
    wind_volume: float = 0.4
    road_volume: float = 0.3
    vehicle_volume: float = 0.6
    horn_volume: float = 0.8
    ambient_volume: float = 0.2

    # Engine sound parameters
    idle_rpm: float = 800.0
    redline_rpm: float = 6500.0
    rpm_smoothing: float = 0.1  # Interpolation factor for RPM changes

    # Spatial audio
    max_audible_distance: float = 100.0  # meters
    doppler_factor: float = 0.3

    # Performance
    update_rate_hz: float = 30.0
    num_channels: int = 16

    def to_dict(self) -> Dict:
        return {
            "master_volume": self.master_volume,
            "enabled": self.enabled,
            "engine_volume": self.engine_volume,
            "exhaust_volume": self.exhaust_volume,
            "wind_volume": self.wind_volume,
            "road_volume": self.road_volume,
            "vehicle_volume": self.vehicle_volume,
            "horn_volume": self.horn_volume,
            "ambient_volume": self.ambient_volume,
            "idle_rpm": self.idle_rpm,
            "redline_rpm": self.redline_rpm,
        }


class EngineSoundGenerator:
    """
    Generates engine sounds by modulating base samples based on RPM.

    Strategy:
    1. Load base engine samples at different RPM ranges (idle, low, mid, high, redline)
    2. Crossfade between samples based on current RPM
    3. Adjust playback speed (pitch) for fine-tuning within each range
    4. Add exhaust pops/crackles on deceleration
    """

    # Default sample files (these should be provided)
    DEFAULT_SAMPLES = {
        "idle": "./audio/engine_idle.wav",
        "low": "./audio/engine_low.wav",
        "mid": "./audio/engine_mid.wav",
        "high": "./audio/engine_high.wav",
        "redline": "./audio/engine_redline.wav",
        "exhaust_pop": "./audio/exhaust_pop.wav",
    }

    # RPM ranges for each sample
    RPM_RANGES = {
        "idle": (0, 1200),
        "low": (1000, 2500),
        "mid": (2200, 4000),
        "high": (3500, 5500),
        "redline": (5000, 7000),
    }

    def __init__(self, config: AudioConfig):
        self.config = config
        self.samples: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.current_rpm = config.idle_rpm
        self.target_rpm = config.idle_rpm
        self._load_samples()

    def _load_samples(self):
        """Load engine sound samples."""
        for name, path in self.DEFAULT_SAMPLES.items():
            try:
                self.samples[name] = pygame.mixer.Sound(path)
                logging.info(f"Loaded engine sample: {name}")
            except Exception as e:
                logging.warning(f"Could not load engine sample {name} from {path}: {e}")
                self.samples[name] = None

    def _generate_synthetic_engine(self, rpm: float, duration_ms: int = 100) -> Optional[pygame.mixer.Sound]:
        """
        Generate synthetic engine sound if samples not available.
        Uses additive synthesis with harmonics based on RPM.
        """
        try:
            sample_rate = 44100
            num_samples = int(sample_rate * duration_ms / 1000)

            # Base frequency from RPM (engine fires at RPM/60 * cylinders/2 for 4-stroke)
            # Assume 4-cylinder: 2 fires per revolution
            base_freq = (rpm / 60) * 2

            t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

            # Additive synthesis with harmonics
            wave = np.zeros(num_samples, dtype=np.float32)

            # Fundamental and harmonics (engine has strong even harmonics)
            harmonics = [1, 2, 3, 4, 6, 8]
            amplitudes = [1.0, 0.5, 0.25, 0.15, 0.1, 0.05]

            for h, a in zip(harmonics, amplitudes):
                freq = base_freq * h
                # Add some frequency wobble for realism
                wobble = np.sin(2 * np.pi * 5 * t) * 0.02
                wave += a * np.sin(2 * np.pi * freq * (1 + wobble) * t)

            # Normalize
            wave = wave / np.max(np.abs(wave)) * 0.8

            # Add some noise for roughness
            noise = np.random.randn(num_samples).astype(np.float32) * 0.05
            wave += noise

            # Clip and convert to int16
            wave = np.clip(wave, -1, 1)
            wave_int = (wave * 32767).astype(np.int16)

            # Stereo
            stereo = np.column_stack([wave_int, wave_int])

            return pygame.mixer.Sound(buffer=stereo)

        except Exception as e:
            logging.error(f"Error generating synthetic engine: {e}")
            return None

    def update(self, engine_state: EngineState) -> Optional[pygame.mixer.Sound]:
        """
        Update engine sound based on current state.
        Returns a Sound object to play or None.
        """
        self.target_rpm = engine_state.rpm

        # Smooth RPM changes
        self.current_rpm += (self.target_rpm - self.current_rpm) * self.config.rpm_smoothing

        # Determine which sample to use based on RPM
        current_sample = None
        for name, (low, high) in self.RPM_RANGES.items():
            if low <= self.current_rpm <= high:
                current_sample = self.samples.get(name)
                break

        if current_sample is None:
            # Generate synthetic if no samples
            return self._generate_synthetic_engine(self.current_rpm)

        return current_sample


class ProximitySoundManager:
    """
    Manages sounds from nearby vehicles and pedestrians.

    Features:
    - Distance-based volume attenuation
    - Doppler effect for approaching/receding vehicles
    - Random honking behaviors in traffic
    - Pedestrian ambient sounds
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.sound_sources: Dict[int, SoundSource] = {}  # actor_id -> SoundSource

    def update_nearby_actors(self, actors: List[Dict]):
        """
        Update sound sources from nearby CARLA actors.

        actors: List of dicts with keys:
            - id: actor ID
            - type: "vehicle" or "pedestrian"
            - position: (x, y, z)
            - velocity: (vx, vy, vz)
            - distance: distance from player
        """
        active_ids = set()

        for actor in actors:
            actor_id = actor.get('id')
            if actor_id is None:
                continue

            active_ids.add(actor_id)
            distance = actor.get('distance', 999)

            if distance > self.config.max_audible_distance:
                # Too far, remove if exists
                if actor_id in self.sound_sources:
                    self._stop_source(actor_id)
                continue

            # Calculate volume based on distance
            volume = self._calculate_volume(distance)

            # Calculate pitch shift (Doppler effect)
            pitch = self._calculate_doppler(actor)

            if actor_id not in self.sound_sources:
                # Create new sound source
                category = SoundCategory.VEHICLE if actor['type'] == 'vehicle' else SoundCategory.PEDESTRIAN
                self.sound_sources[actor_id] = SoundSource(
                    category=category,
                    position=actor.get('position', (0, 0, 0)),
                    velocity=actor.get('velocity', (0, 0, 0)),
                    volume=volume,
                    pitch=pitch,
                )
            else:
                # Update existing
                source = self.sound_sources[actor_id]
                source.position = actor.get('position', source.position)
                source.velocity = actor.get('velocity', source.velocity)
                source.volume = volume
                source.pitch = pitch

        # Remove sources for actors no longer nearby
        for actor_id in list(self.sound_sources.keys()):
            if actor_id not in active_ids:
                self._stop_source(actor_id)

    def _calculate_volume(self, distance: float) -> float:
        """Calculate volume based on distance (inverse square law with falloff)."""
        if distance <= 1:
            return 1.0
        # Inverse square with minimum
        return max(0.0, 1.0 / (distance * 0.1) ** 2)

    def _calculate_doppler(self, actor: Dict) -> float:
        """Calculate pitch shift due to Doppler effect."""
        # Simplified Doppler: pitch increases when approaching
        velocity = actor.get('velocity', (0, 0, 0))
        player_dir = actor.get('direction_to_player', (0, 0, 0))

        # Relative velocity towards player
        rel_vel = sum(v * d for v, d in zip(velocity, player_dir))

        # Speed of sound ~343 m/s
        speed_of_sound = 343.0
        doppler_shift = (speed_of_sound + rel_vel * self.config.doppler_factor) / speed_of_sound

        return np.clip(doppler_shift, 0.5, 2.0)

    def _stop_source(self, actor_id: int):
        """Stop and remove a sound source."""
        if actor_id in self.sound_sources:
            source = self.sound_sources[actor_id]
            if source.channel:
                source.channel.stop()
            del self.sound_sources[actor_id]


class AmbientSoundManager:
    """
    Manages ambient environmental sounds.

    Includes:
    - Wind noise (based on speed)
    - Road/tire noise (based on speed and surface)
    - Weather sounds (rain, thunder)
    - City ambient noise
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.wind_volume = 0.0
        self.road_volume = 0.0

    def update(self, speed_kmh: float, weather: str = "clear", surface: str = "asphalt"):
        """Update ambient sound levels based on driving conditions."""
        # Wind noise increases with speed (logarithmic)
        if speed_kmh > 10:
            self.wind_volume = min(1.0, math.log10(speed_kmh) / 2.5) * self.config.wind_volume
        else:
            self.wind_volume = 0.0

        # Road noise increases with speed
        self.road_volume = min(1.0, speed_kmh / 120) * self.config.road_volume

        # Weather effects
        # TODO: Add rain/thunder sounds based on weather parameter


class AudioEngine:
    """
    Main audio engine for Q-DRIVE Alpha.

    Coordinates all audio subsystems:
    - Engine sounds
    - Proximity sounds (vehicles, pedestrians)
    - Ambient sounds
    - UI sounds (blinkers, warnings)
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Current state
        self.engine_state = EngineState()
        self.player_position = (0, 0, 0)
        self.nearby_actors: List[Dict] = []

        # Subsystems (lazy initialized)
        self._engine_gen: Optional[EngineSoundGenerator] = None
        self._proximity: Optional[ProximitySoundManager] = None
        self._ambient: Optional[AmbientSoundManager] = None

        # Pygame mixer state
        self._mixer_initialized = False

    def _init_mixer(self):
        """Initialize pygame mixer for audio."""
        if self._mixer_initialized:
            return

        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(self.config.num_channels)
            self._mixer_initialized = True
            logging.info("Audio mixer initialized")
        except Exception as e:
            logging.error(f"Failed to initialize audio mixer: {e}")
            self.config.enabled = False

    def _init_subsystems(self):
        """Initialize audio subsystems."""
        self._engine_gen = EngineSoundGenerator(self.config)
        self._proximity = ProximitySoundManager(self.config)
        self._ambient = AmbientSoundManager(self.config)

    def start(self):
        """Start the audio engine."""
        if not self.config.enabled:
            logging.info("Audio engine disabled")
            return

        self._init_mixer()
        if not self._mixer_initialized:
            return

        self._init_subsystems()
        self.running = True

        # Start update thread
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        logging.info("Audio engine started")

    def stop(self):
        """Stop the audio engine."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        pygame.mixer.quit()
        self._mixer_initialized = False
        logging.info("Audio engine stopped")

    def _audio_loop(self):
        """Main audio update loop."""
        update_interval = 1.0 / self.config.update_rate_hz

        while self.running:
            try:
                with self._lock:
                    self._update_audio()
                time.sleep(update_interval)
            except Exception as e:
                logging.error(f"Audio engine error: {e}")

    def _update_audio(self):
        """Update all audio systems."""
        if not self._mixer_initialized:
            return

        # Update engine sound
        if self._engine_gen:
            sound = self._engine_gen.update(self.engine_state)
            # Note: Actual playback logic would go here

        # Update proximity sounds
        if self._proximity:
            self._proximity.update_nearby_actors(self.nearby_actors)

        # Update ambient sounds
        if self._ambient:
            self._ambient.update(
                speed_kmh=self.engine_state.speed_kmh,
                weather="clear"
            )

    def update(self, **kwargs):
        """
        Update audio engine with current simulation state.

        Kwargs:
            rpm: Engine RPM
            speed_kmh: Vehicle speed
            throttle: Throttle position (0-1)
            gear: Current gear
            acceleration: Current acceleration
            position: Player position (x, y, z)
            nearby_actors: List of nearby actor dicts
        """
        with self._lock:
            self.engine_state.rpm = kwargs.get('rpm', self.engine_state.rpm)
            self.engine_state.speed_kmh = kwargs.get('speed_kmh', self.engine_state.speed_kmh)
            self.engine_state.throttle = kwargs.get('throttle', self.engine_state.throttle)
            self.engine_state.gear = kwargs.get('gear', self.engine_state.gear)
            self.engine_state.acceleration = kwargs.get('acceleration', self.engine_state.acceleration)

            self.player_position = kwargs.get('position', self.player_position)
            self.nearby_actors = kwargs.get('nearby_actors', self.nearby_actors)

    def play_horn(self, duration_ms: int = 500):
        """Play horn sound."""
        # TODO: Implement horn sound
        pass

    def play_blinker(self, on: bool = True):
        """Play blinker click sound."""
        # TODO: Implement blinker sound
        pass

    def play_collision(self, intensity: float = 1.0):
        """Play collision sound based on intensity."""
        # TODO: Implement collision sound
        pass

    def set_master_volume(self, volume: float):
        """Set master volume (0-1)."""
        self.config.master_volume = max(0.0, min(1.0, volume))

    def enable(self):
        """Enable audio."""
        self.config.enabled = True
        self.start()

    def disable(self):
        """Disable audio."""
        self.config.enabled = False
        self.stop()


def main():
    """Test audio engine standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Q-DRIVE Audio Engine Test")
    parser.add_argument("--test", action="store_true", help="Run audio test")
    args = parser.parse_args()

    if args.test:
        config = AudioConfig(enabled=True)
        engine = AudioEngine(config)
        engine.start()

        print("Audio engine running. Press Ctrl+C to stop.")
        try:
            # Simulate RPM changes
            rpm = 800
            direction = 1
            while True:
                engine.update(rpm=rpm, speed_kmh=rpm/50, throttle=0.5, gear=3)
                rpm += direction * 100
                if rpm > 6000:
                    direction = -1
                elif rpm < 800:
                    direction = 1
                time.sleep(0.1)
        except KeyboardInterrupt:
            engine.stop()
            print("Audio engine stopped.")


if __name__ == "__main__":
    main()
