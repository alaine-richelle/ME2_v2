# Raspberry Pi 5 + Ubuntu: boot media to ME2 dashboard

This is the Ubuntu alternative to `RASPBERRY_PI_5_WORKFLOW.md`. The reference
installation here is **Ubuntu Server 24.04 LTS, 64-bit, preinstalled for
Raspberry Pi**. Ubuntu supports this board/release combination. Use a current
24.04 point image from Canonical's Pi image listings; record its exact name
and SHA-256. This choice is a stable baseline for this project, not a claim
that 24.04 is the newest Ubuntu release. [U1]

The existing Python inference pipeline, `.me2` bundles, dashboard and systemd
unit can be used with this Ubuntu setup. OS and target-hardware verification
remain necessary. No Ubuntu image was flashed and no physical Pi was tested
while preparing this guide.

## Workflow

```mermaid
flowchart LR
  A[Select trained model bundles] --> B[Select Ubuntu Pi ARM64 image]
  B --> C[Imager catalog or Use custom / flash / verify]
  C --> D[Imager customization or cloud-init]
  D --> E[First boot and SSH]
  E --> F[Ubuntu packages and Python environment]
  F --> G[USB microphone and model checks]
  G --> H[Dashboard demo then live inference]
  H --> I[Save intent-to-action mappings]
  I --> J[systemd startup and Pi acceptance tests]
```

## 1. Choose Ubuntu in Raspberry Pi Imager

Prepare the Pi 5, cooling, suitable power supply, USB microphone, microSD
card/reader, and Ethernet or Wi-Fi. For this project, budget at least 32 GB
of storage for the OS, environment and logs.

In Raspberry Pi Imager, select **Raspberry Pi 5**, then use:

```text
Choose OS → Other general-purpose OS → Ubuntu → Ubuntu Server 24.04 LTS (64-bit)
```

This is the recommended path for the requested Ubuntu choice: Imager obtains
the matching Raspberry Pi image itself. For an attached monitor and local
graphical session, choose the corresponding Ubuntu Desktop 24.04 LTS
(64-bit) entry and follow its first-boot setup wizard. The Server path below
needs no desktop: use your workstation's browser for the dashboard. Imager
labels and offered point releases can change; choose a Pi-specific ARM64
Ubuntu Server entry rather than a PC ISO. [U1, U2]

If by “custom OS” you mean Imager’s **Use custom** option, that is a separate
path for a locally downloaded image. Use the current Pi-specific file ending
in `-preinstalled-server-arm64+raspi.img.xz`; an AMD64 PC ISO or generic ARM
server ISO does not follow this installation path. For a privately rebuilt
image, confirm Pi 5 kernel, firmware, boot files and cloud-init support with
its producer.

Use conventional Ubuntu Server/Desktop for this guide. Ubuntu Core uses a
different application packaging/provisioning model; these `apt`, virtualenv
and source-checkout instructions are not an Ubuntu Core deployment. [U6]

## 2. Flash the Ubuntu Server image

### A. Ubuntu chosen from Imager’s catalog

1. In Imager, select the Ubuntu Server choice from step 1 and the intended
   microSD card.
2. If the OS customization dialog is offered and you use it, set hostname
   `ateko-pi`, timezone `Asia/Manila`, a `me2` administrator, network details
   and SSH access there.
3. Write and complete verification. Writing erases the selected card.
4. Use either those Imager settings **or** the manual cloud-init files in
   step 3. Do not overwrite Imager-created provisioning files with a second,
   conflicting configuration.

### B. Locally downloaded Ubuntu image with “Use custom”

On the workstation:

1. Download the selected Pi image and its matching published checksum.
2. Compare the image's digest against that checksum. On macOS:

   ```sh
   shasum -a 256 "/path/to/the-downloaded-ubuntu-pi-image.img.xz"
   ```

3. Open Raspberry Pi Imager; select **Raspberry Pi 5**.
4. Choose **Use custom** and select the downloaded `.img.xz`. Imager accepts
   compressed disk images. [U2]
5. Select the intended microSD card. Writing erases its contents.
6. Write and complete verification. Reinsert/remount the card on the
   workstation if Imager ejects it.

For this downloaded-image path, use the explicit cloud-init method below
rather than assuming that every Imager customization option was applied.

## 3. Manual headless provisioning with cloud-init

On the card's FAT boot partition, normally labeled `system-boot`, Ubuntu Pi
images carry the cloud-init seed: `user-data`, `network-config`, and
`meta-data`. Use plain text files with these exact names, without `.txt`.
Keep the image's existing `meta-data`. [U3]

Use this step only when you did not use Imager's compatible OS-customization
dialog, or when using the downloaded image through **Use custom**. Repository
templates:

```text
deploy/raspberry-pi/ubuntu/user-data.example
deploy/raspberry-pi/ubuntu/network-config.ethernet.example
deploy/raspberry-pi/ubuntu/network-config.wifi.example
```

Copy the user template to the partition as **`user-data`**. Replace
`REPLACE_WITH_YOUR_COMPLETE_SSH_PUBLIC_KEY` with your complete existing
public key, for example the line from `~/.ssh/id_ed25519.pub`. Do not copy
the private key. The resulting shape is:

```yaml
#cloud-config
hostname: ateko-pi
manage_etc_hosts: true
timezone: Asia/Manila
disable_root: true
ssh_pwauth: false
users:
  - name: me2
    gecos: ME2 administrator
    shell: /bin/bash
    groups: [adm, sudo, audio]
    lock_passwd: true
    sudo: ["ALL=(ALL) NOPASSWD:ALL"]
    ssh_authorized_keys:
      - "YOUR_COMPLETE_PUBLIC_KEY_LINE"
```

This creates an SSH-key-only bootstrap administrator named `me2`, with
passwordless sudo for installing the system. Its home and primary group
match the existing service template. These user, sudo and authorized-key
settings are supported cloud-init options. [U4]

Copy **one** network template to the partition as **`network-config`**:

- Ethernet: use `network-config.ethernet.example`, then connect Ethernet
  before powering on.
- Wi-Fi: use `network-config.wifi.example`; replace the SSID/password and
  preserve YAML indentation. This assumes the standard Pi image's `eth0`
  and `wlan0` interface names; adapt them for a privately rebuilt image.

The seed file starts at `version: 2`; a live `/etc/netplan/*.yaml` file instead
has a top-level `network:` wrapper. Configure the connection actually
available on first boot. [U3, U5]

The templates contain placeholders and are not ready to boot unchanged.
Keep filled-in network credentials off the repository. Safely eject after
editing, install the card in the Pi, connect the microphone/network/cooling,
and power on.

If a custom image does not support this seed, use a monitor/keyboard and its
documented first-boot setup to create `me2` and enable SSH instead. On a stock
Server image left at defaults, complete the local default-user/password
change flow first; do not assume default-password SSH login is enabled. [U1, U3]

## 4. First boot, SSH and Ubuntu packages

Wait for cloud-init to complete account and host-key creation. Find the Pi's
IP in your router's DHCP list. The examples use `192.168.1.50` as a
**placeholder**; substitute the Pi's real address everywhere. Initial setup
uses the address because `.local` discovery is not assumed. [U3]

From the workstation:

```sh
ssh me2@192.168.1.50
```

On the Pi:

```sh
cloud-init status --wait
cat /etc/os-release
uname -m
dpkg --print-architecture
hostname -I
id
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-pip libportaudio2 libsndfile1 \
  alsa-utils rsync openssh-server avahi-daemon
sudo hostnamectl set-hostname ateko-pi
sudo timedatectl set-timezone Asia/Manila
sudo usermod -aG audio me2
sudo systemctl enable --now ssh
sudo systemctl enable --now avahi-daemon
sudo reboot
```

Expect Ubuntu, `aarch64`, `arm64`, and the `me2` account. Investigate any
cloud-init errors before continuing. After reboot, reconnect; verify
`id -nG` contains `audio`. PortAudio is the Linux library used by
`sounddevice`. [U7]

After installing Avahi, `ateko-pi.local` can be used if the workstation and
network support mDNS; the IP remains a fallback. [U3]

If first boot networking fails, inspect `cloud-init status --long`,
`/var/log/cloud-init.log`, `/var/log/cloud-init-output.log`, `ip -br link`,
and `sudo netplan get` from a local console. Match the observed interface
names. Use `sudo netplan try` for later network edits so an unconfirmed
change can roll back. Boot-seed edits are first-boot provisioning, not a
general way to reconfigure an already provisioned machine. [U3, U5]

## 5. Transfer ME2 and the selected model bundles

On the workstation, from this repository root:

```sh
ssh me2@192.168.1.50 \
  'mkdir -p ~/ME2_v2/wake-word/models ~/ME2_v2/vcm/models ~/ME2_v2/fixtures'
rsync -av --exclude '__pycache__' --exclude '*.egg-info' \
  pyproject.toml README.md src config docs deploy \
  me2@192.168.1.50:/home/me2/ME2_v2/
rsync -av wake-word/models/wake_word_v5_regularized_seed43.me2 \
  me2@192.168.1.50:/home/me2/ME2_v2/wake-word/models/
rsync -av vcm/models/vcm_ctc_model.me2 \
  me2@192.168.1.50:/home/me2/ME2_v2/vcm/models/
shasum -a 256 wake-word/models/wake_word_v5_regularized_seed43.me2 \
  vcm/models/vcm_ctc_model.me2 > deployment-models.sha256
scp deployment-models.sha256 me2@192.168.1.50:/home/me2/ME2_v2/
```

These filenames match `config/dashboard.json` and the current engineering
fixtures. For a selected approved release, substitute its exact bundle
filenames and update both configuration paths before transferring. Copy
reviewed recordings as `fixtures/wake.wav` and `fixtures/command.wav`.

On the Pi:

```sh
cd /home/me2/ME2_v2
sha256sum -c deployment-models.sha256
mkdir -p logs results dashboard-state
```

Recreate the Python environment on Ubuntu; the macOS `.venv` is not a
deployment artifact. No dataset download or retraining is needed to run the
existing dashboard with the selected bundles.

## 6. Install and check CPU inference

On the Pi:

```sh
cd /home/me2/ME2_v2
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: torch
.venv/bin/python -m pip install -e '.[microphone]'
.venv/bin/python -c \
  'import platform, torch; print(platform.machine(), torch.__version__); print(torch.ones(2) + 1)'
.venv/bin/python -m pip freeze > results/pi5-ubuntu-environment.txt
```

If Torch has no compatible binary, check the official stable Linux/Pip/CPU
installation guidance and use an ARM64/Python combination with a published
wheel. Record the actual version installed. Do not install a CUDA build for
this Pi CPU deployment. [U8]

## 7. Verify the microphone and both models

On the Pi:

```sh
arecord -l
.venv/bin/python -m sounddevice
```

Select an input-capable PortAudio device. A PortAudio index is not an ALSA
card number. Set the device's unique name as `input_device` in
`config/dashboard.json`, then test the exact live format:

```sh
.venv/bin/python - <<'PY'
import sounddevice as sd
import soundfile as sf

device = "YOUR INPUT DEVICE NAME"
sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=16000)
audio = sd.rec(3 * 16000, samplerate=16000, channels=1, dtype="float32", device=device)
sd.wait()
sf.write("fixtures/microphone-check.wav", audio, 16000)
print(sd.query_devices(device, kind="input"))
PY
```

Listen to the recording. The present live pipeline requests 16 kHz mono;
a microphone accepting only 48 kHz needs a tested conversion device or
capture adapter. The browser uses audio from the Pi's microphone.
`sounddevice` supplies device enumeration and format validation. [U9]

For Ubuntu Server, use a microphone device accessible through ALSA by the
`me2` account and the boot service. On Ubuntu Desktop, verify the chosen
device in the same execution context as the service: a graphical session's
audio device may depend on that session's sound server. Do not assume a
desktop recording test establishes access for an unattended system service.

Run the existing file check:

```sh
.venv/bin/python -m me2.app --interactive \
  --wake-model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --vcm-model vcm/models/vcm_ctc_model.me2 \
  --wake-audio fixtures/wake.wav --command-audio fixtures/command.wav
```

This checks the bundles and a simple file wake gate, not continuous-stream
quality. See the original workflow's microphone, benchmark and acceptance
sections for the full sequence of positive/negative tests.

## 8. Run the dashboard and connect from the workstation

First, on the Pi:

```sh
.venv/bin/python -m me2.dashboard --demo --autostart
```

In another workstation terminal:

```sh
ssh -N -L 8765:127.0.0.1:8765 me2@192.168.1.50
```

Open `http://127.0.0.1:8765` on the workstation. Stop any workstation server
already using that port. Both tunnel ends must use the dashboard's configured
port, since the application validates the HTTP host.

Send an accepted `light_on` example, try a rejection, disable/save a mapping,
and verify stop/restart. The demo uses synthetic predictions. Once checked,
stop it with Ctrl+C and launch the live path on the Pi:

```sh
.venv/bin/python -m me2.dashboard --config config/dashboard.json
```

Press **Start listening**. The integrated flow is:

```text
USB mic → complete “ateko” wake detection → command capture
        → VCM decoding + rejection → enabled intent mapping
        → dashboard action → cooldown → wake listening
```

As recorded on 2026-10-06, the bundled wake fixture is explicitly
`deployment_ready=false`. It blocks live listening until an approved model
is selected. For a supervised engineering microphone experiment, the
existing explicit override is:

```sh
.venv/bin/python -m me2.dashboard \
  --config config/dashboard.json --allow-unverified-wake-model
```

Ubuntu does not change this readiness requirement. It also does not turn
synthetic model-validation results into real-speaker deployment evidence.

## 9. Map recognized commands in the UI

The existing dashboard contains all 19 intent rows and shows which ones
exist in the loaded model:

| Intent | Default action |
| --- | --- |
| `light_on`, `light_off` | Simulated light |
| `play_music`, `pause`, `stop`, `next` | Simulated player |
| `volume_up`, `volume_down` | Simulated volume |
| Other supported intents | Display recognition and returned slots |

Choose an action or disable the mapping, then select **Save mappings**.
Settings persist in `dashboard-state/mappings.json`. Rejected commands cannot
dispatch actions. Missing slots remain missing; the selected legacy grammar
does not supply arbitrary numbers, times, colors or contact names.

Physical lights, media players, timers, calls and messages still need actual
action handlers. This UI currently updates simulated devices.

## 10. Start automatically on Ubuntu

The existing systemd unit can be reused with the `me2` account and
`/home/me2/ME2_v2` directory created above. Review its paths if using another
username. Run these commands on the Pi after stopping the foreground server:

```sh
cd /home/me2/ME2_v2
mkdir -p logs dashboard-state
sudo cp deploy/raspberry-pi/me2-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now me2-dashboard.service
systemctl status me2-dashboard.service --no-pager
journalctl -u me2-dashboard.service -n 80 --no-pager
```

It uses the account's audio group, starts the worker with `--autostart`,
binds HTTP to loopback, and writes logs/mappings in the allowed directories.
It has no unverified-model override. With the current preliminary wake
artifact, the web service can run while the UI reports the readiness error.

Test microphone access and restart/reboot behavior in this service context.
A worker error remains visible in the UI; systemd process restart does not
by itself fix a model or microphone configuration error.

## 11. Ubuntu measurements and release checks

Benchmark both models on the actual Pi:

```sh
.venv/bin/python -m me2.benchmark fixtures/wake.wav \
  --model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --iterations 200 --threads 2 --output results/pi5_ubuntu_wake_benchmark.json
.venv/bin/python -m me2.benchmark fixtures/command.wav \
  --model vcm/models/vcm_ctc_model.me2 \
  --iterations 200 --threads 2 --output results/pi5_ubuntu_vcm_benchmark.json
```

For temperature, inspect available Linux thermal zones rather than assuming
`vcgencmd` is installed by Ubuntu:

```sh
for zone in /sys/class/thermal/thermal_zone*; do
  test -r "$zone/type" && cat "$zone/type"
  test -r "$zone/temp" && cat "$zone/temp"
done
```

The sysfs `temp` value is in millidegrees Celsius; identify the CPU/SoC zone
from its `type`. This is a temperature reading, not a measurement of historical
throttling or undervoltage. [U10]

The present dashboard reads thermal zone 0 when available; confirm that
zone's identity on Ubuntu before interpreting its widget as CPU temperature.
Use platform-supported firmware/kernel tools for throttling evidence. Do not
assume Raspberry Pi OS packages, `raspi-config`, or firmware-update recipes
apply unchanged to Ubuntu.

Before deployment, retain: OS image/hash, kernel and Python/Torch versions,
both model hashes, microphone name, calibrated policy, Pi benchmark results,
continuous negative-audio and real-speaker results, reboot/service tests,
and sustained CPU/RSS/thermal measurements. Test command-without-wake, wake
without-command, out-of-scope speech, disabled mappings, microphone recovery
and stopping during decoding.

Use the original workflow's release/rollback process with the Ubuntu
environment report. No new Ubuntu hardware validation is claimed here.

## References

Platform documentation checked 2026-10-06. Repository behavior is described
from the implemented `me2.dashboard`, `me2.app`, and `me2.streaming` modules.

- U1 — Canonical's Raspberry Pi image/install guidance: `https://ubuntu.com/hardware/docs/boards/how-to/ubuntu_supported/raspberry-pi/`
- U2 — Ubuntu's Raspberry Pi Imager manual: `https://manpages.ubuntu.com/manpages/noble/man1/rpi-imager.1.html`
- U3 — Canonical headless provisioning: `https://ubuntu.com/hardware/docs/boards/how-to/special_hardware/headless-usage/`
- U4 — cloud-init user and SSH examples: `https://cloudinit.readthedocs.io/en/latest/reference/examples.html`
- U5 — Netplan examples and tools: `https://netplan.readthedocs.io/en/stable/`
- U6 — Ubuntu Core installation: `https://documentation.ubuntu.com/core/tutorials/try-pre-built-images/use-raspberry-pi-imager/`
- U7 — sounddevice installation: `https://python-sounddevice.readthedocs.io/en/latest/installation.html`
- U8 — PyTorch installation: `https://docs.pytorch.org/get-started/locally/`
- U9 — sounddevice hardware checks: `https://python-sounddevice.readthedocs.io/en/latest/api/checking-hardware.html`
- U10 — Linux thermal sysfs documentation: `https://www.kernel.org/doc/html/latest/driver-api/thermal/sysfs-api.html`
