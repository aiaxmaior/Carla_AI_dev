"""
Core.Audio - Experimental Audio Engine Module

Provides a modular audio framework for introducing realistic driving sounds.
EXPERIMENTAL: Activates only when --enable-audio is passed.

Features:
- Engine sound modulation based on RPM, acceleration
- Vehicle proximity sounds (other cars, pedestrians)
- Ambient environmental sounds (wind, road noise)
- Honking behaviors
- Collision sounds

Usage:
    from Core.Audio import AudioEngine, AudioConfig

    config = AudioConfig(enabled=True)
    audio = AudioEngine(config)
    audio.start()

    # In game loop:
    audio.update(rpm=3500, speed_kmh=60, throttle=0.5, ...)

    # Cleanup:
    audio.stop()
"""

from .AudioEngine import (
    AudioEngine,
    AudioConfig,
    EngineState,
    SoundCategory,
    SoundSource,
    EngineSoundGenerator,
    ProximitySoundManager,
    AmbientSoundManager,
)

__all__ = [
    "AudioEngine",
    "AudioConfig",
    "EngineState",
    "SoundCategory",
    "SoundSource",
    "EngineSoundGenerator",
    "ProximitySoundManager",
    "AmbientSoundManager",
]
