"""Stable, privacy-safe selection of ALSA capture and playback endpoints."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

DeviceKind = Literal["capture", "playback"]
_DEVICE_LINE = re.compile(
    r"^\s*card\s+(?P<card>\d+):\s*(?P<card_id>[^\s\[]+)\s+"
    r"\[(?P<card_name>[^\]]+)\],\s*device\s+(?P<device>\d+):\s*"
    r"(?P<device_name>[^\[]+)"
)
_EXPLICIT_ALSA = re.compile(r"^(?:default|sysdefault|front|dsnoop|plughw|hw):?.*$")


class AudioDeviceError(RuntimeError):
    """A concise, non-sensitive device-selection failure."""

    def __init__(self, code: str, *, kind: DeviceKind) -> None:
        self.code = code
        self.kind = kind
        descriptions = {
            "no_device": f"no_{kind}_device; connect a supported device and retry",
            "ambiguous": (
                f"ambiguous_{kind}_device; set a stable selector such as "
                f"name:USB Audio in configuration"
            ),
            "busy": f"{kind}_device_busy; close the current audio client or use the shared default backend",
            "permission": f"{kind}_device_permission_denied; confirm audio-group membership and log in again",
            "format": f"{kind}_device_format_unsupported; require 16-kHz mono PCM for microphone input",
            "unavailable": f"{kind}_device_unavailable; re-enumerate ALSA devices and check the cable",
        }
        super().__init__(descriptions.get(code, f"{kind}_device_unavailable"))


@dataclass(frozen=True)
class AlsaDevice:
    kind: DeviceKind
    card_id: str
    card_name: str
    device: int
    device_name: str

    @property
    def stable_pcm(self) -> str:
        return f"plughw:CARD={self.card_id},DEV={self.device}"

    def as_dict(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "card_id": self.card_id,
            "card_name": self.card_name,
            "device": self.device,
            "device_name": self.device_name,
            "stable_pcm": self.stable_pcm,
        }


@dataclass(frozen=True)
class CaptureSelection:
    backend: str
    input_device: str | int | None
    alsa_device: str | None
    selection_source: str


def list_alsa_devices(kind: DeviceKind, *, runner=subprocess.run) -> list[AlsaDevice]:
    executable = shutil.which("arecord" if kind == "capture" else "aplay")
    if executable is None:
        raise AudioDeviceError("no_device", kind=kind)
    result = runner(
        [executable, "-l"], capture_output=True, text=True, timeout=5, check=False
    )
    if result.returncode:
        raise AudioDeviceError("no_device", kind=kind)
    devices = []
    for line in result.stdout.splitlines():
        matched = _DEVICE_LINE.match(line)
        if matched is None:
            continue
        devices.append(AlsaDevice(
            kind=kind,
            card_id=matched["card_id"],
            card_name=matched["card_name"].strip(),
            device=int(matched["device"]),
            device_name=matched["device_name"].strip(),
        ))
    if not devices:
        raise AudioDeviceError("no_device", kind=kind)
    return devices


def resolve_alsa_device(
    kind: DeviceKind,
    selector: str = "auto",
    *,
    runner=subprocess.run,
) -> tuple[str, str]:
    """Resolve ``auto``/``name:...`` selectors without persisting card numbers."""
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("audio device selector must be a non-empty string")
    selector = selector.strip()
    if selector == "default":
        return selector, "shared_default"
    if selector not in {"auto"} and not selector.startswith("name:"):
        if not _EXPLICIT_ALSA.fullmatch(selector) or re.search(r"(?:plughw|hw):\d+,\d+", selector):
            raise ValueError("use auto, name:selector, default, or a named ALSA PCM")
        return selector, "explicit_alsa"

    candidates = list_alsa_devices(kind, runner=runner)
    if selector == "auto":
        matches = candidates
    else:
        term = selector.removeprefix("name:").strip().casefold()
        if not term:
            raise ValueError("name: audio selector requires a non-empty stable name fragment")
        matches = [
            item for item in candidates
            if term in f"{item.card_id} {item.card_name} {item.device_name}".casefold()
        ]
    if not matches:
        raise AudioDeviceError("no_device", kind=kind)
    if len(matches) != 1:
        raise AudioDeviceError("ambiguous", kind=kind)
    return matches[0].stable_pcm, "named_alsa"


def select_capture_input(
    audio_backend: str,
    *,
    input_device: str | int | None,
    alsa_device: str,
    sample_rate: int,
) -> CaptureSelection:
    """Prefer the shared system default for ``auto`` before direct ALSA fallback."""
    if audio_backend not in {"auto", "sounddevice", "arecord"}:
        raise ValueError("audio_backend must be auto, sounddevice, or arecord")
    if audio_backend in {"auto", "sounddevice"}:
        try:
            import sounddevice as sd

            sd.check_input_settings(
                device=input_device, channels=1, dtype="float32", samplerate=sample_rate
            )
            return CaptureSelection(
                backend="sounddevice", input_device=input_device, alsa_device=None,
                selection_source="shared_default" if input_device is None else "explicit_sounddevice",
            )
        except Exception as error:
            if audio_backend == "sounddevice" or input_device is not None:
                raise AudioDeviceError("unavailable", kind="capture") from error
    device, source = resolve_alsa_device("capture", alsa_device)
    return CaptureSelection(
        backend="arecord", input_device=None, alsa_device=device, selection_source=source
    )


def validate_alsa_device(kind: DeviceKind, device: str, *, runner=subprocess.run) -> None:
    """Open a PCM briefly for format validation without recording or playing content."""
    executable = shutil.which("arecord" if kind == "capture" else "aplay")
    if executable is None:
        raise AudioDeviceError("no_device", kind=kind)
    command = [executable, "-D", device, "--dump-hw-params", "-f", "S16_LE", "-c", "1"]
    if kind == "capture":
        command.extend(["-r", "16000", "-d", "1", "-t", "raw", "/dev/null"])
    else:
        command.extend(["-r", "48000", "-t", "raw", "/dev/null"])
    result = runner(command, capture_output=True, text=True, timeout=5, check=False)
    if result.returncode == 0:
        return
    text = (result.stderr or "").casefold()
    if "busy" in text:
        raise AudioDeviceError("busy", kind=kind)
    if "permission denied" in text:
        raise AudioDeviceError("permission", kind=kind)
    if "format" in text or "rate" in text:
        raise AudioDeviceError("format", kind=kind)
    raise AudioDeviceError("unavailable", kind=kind)
