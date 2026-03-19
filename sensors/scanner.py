"""
sensors/scanner.py
──────────────────
Network utility to auto-discover ESP32-CAMs on the local subnet.

Discovery logic:
  1. Determine the Raspberry Pi's local IP.
  2. Derive the subnet range and exclude: Pi IP, gateway IP, network IP.
  3. Run nmap on the subnet to find devices with an open HTTP port.
  4. Query the /id endpoint on each candidate to identify camera position.
  5. Return a dict mapping { 'left'|'right'|'rear' → stream_url }.
"""

import logging
import os
import socket
import subprocess
from typing import Dict, Optional, Set
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def _get_expected_positions() -> Set[str]:
    try:
        from config import CAMERA_PORTS  # type: ignore
        if isinstance(CAMERA_PORTS, dict) and CAMERA_PORTS:
            return {str(k) for k in CAMERA_PORTS}
    except Exception:
        pass
    return {"left", "right", "rear"}


def _get_candidate_http_ports() -> Set[int]:
    """
    Derive which ports to scan from the configured ESP32-CAM URLs.
    Falls back to {80}, which is the standard ESP32-CAM HTTP port.
    """
    ports: Set[int] = {80}
    try:
        from config import CAMERA_PORTS  # type: ignore
        if isinstance(CAMERA_PORTS, dict):
            for cfg in CAMERA_PORTS.values():
                if not isinstance(cfg, dict):
                    continue
                url = cfg.get("url")
                if not isinstance(url, str) or not url:
                    continue
                try:
                    p = urlparse(url).port
                    if p:
                        ports.add(int(p))
                except Exception:
                    continue
    except Exception:
        pass
    return ports


def get_local_ip() -> str:
    """Returns the primary IP of the Raspberry Pi / host machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_gateway_ip() -> str:
    """Returns the default gateway IP for exclusion from the scan."""
    try:
        if os.name == "nt":
            # Windows
            out = subprocess.check_output(["route", "print", "0.0.0.0"], text=True)
            for line in out.split("\n"):
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "0.0.0.0":
                    return parts[2]
        else:
            # Linux (Raspberry Pi)
            with open("/proc/net/route") as fh:
                for line in fh:
                    fields = line.strip().split()
                    if len(fields) > 3 and fields[1] == "00000000" and int(fields[3], 16) & 2:
                        import struct
                        return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except Exception as exc:
        log.warning("Failed to get gateway IP: %s", exc)
    return ""


def _query_camera_id(ip: str, timeout: float = 3.0) -> Optional[str]:
    """
    Query the /id endpoint on a candidate IP.

    Returns the camera position ('left' | 'right' | 'rear') or None.
    """
    res = None
    try:
        res = requests.get(f"http://{ip}/id", timeout=(timeout, timeout))
        if res.status_code == 200:
            data     = res.json()
            position = data.get("position")
            if position in ("left", "right", "rear"):
                return position
    except Exception:
        pass
    finally:
        try:
            if res is not None:
                res.close()
        except Exception:
            pass
    return None


def discover_esp32_cameras() -> Dict[str, str]:
    """
    Scan the local subnet and return a mapping of
    ``{ position → stream_url }`` for all found ESP32-CAMs.

    Returns an empty dict if discovery fails or nmap is not installed.
    """
    mappings: Dict[str, str] = {}

    local_ip   = get_local_ip()
    gateway_ip = get_gateway_ip()

    if local_ip == "127.0.0.1":
        log.error("[Scanner] Could not determine local IP. Network offline.")
        return mappings

    base_ip       = local_ip.rsplit(".", 1)[0]
    network_ip    = f"{base_ip}.0"
    broadcast_ip  = f"{base_ip}.255"
    target_range  = f"{base_ip}.0/24"

    expected_positions = _get_expected_positions()
    scan_ports         = sorted(_get_candidate_http_ports())
    scan_ports_arg     = ",".join(str(p) for p in scan_ports)

    excludes    = [local_ip, network_ip, broadcast_ip]
    if gateway_ip:
        excludes.append(gateway_ip)
    exclude_str = ",".join(excludes)

    log.info(
        "[Scanner] Host IP: %s | Gateway: %s | Range: %s | Ports: %s | Excluding: %s",
        local_ip, gateway_ip, target_range, scan_ports_arg, exclude_str,
    )

    candidate_ips = []
    try:
        result = subprocess.check_output(
            [
                "nmap", "-p", scan_ports_arg, "--open",
                "--exclude", exclude_str, "-oG", "-", target_range,
            ],
            stderr=subprocess.STDOUT,
            timeout=45,
            text=True,
        )
        for line in result.split("\n"):
            # Grepable nmap format:
            # Host: 192.168.1.10 ()  Ports: 80/open/tcp//http///
            if "Host:" in line and "Ports:" in line and "/open/tcp" in line:
                ip = line.split("Host: ")[1].split(" ")[0]
                candidate_ips.append(ip)

    except FileNotFoundError:
        log.error(
            "[Scanner] nmap not found! "
            "Install it: Linux → sudo apt install nmap | Windows → https://nmap.org"
        )
        return mappings
    except subprocess.TimeoutExpired:
        log.error("[Scanner] nmap scan timed out.")
        return mappings
    except Exception as exc:
        log.error("[Scanner] nmap error: %s", exc)
        return mappings

    log.info("[Scanner] Found %d candidate web server(s): %s", len(candidate_ips), candidate_ips)

    missing = set(expected_positions)
    for ip in candidate_ips:
        if not missing:
            break

        position = _query_camera_id(ip, timeout=3.0)
        if position and position in missing:
            mappings[position] = f"http://{ip}/stream"
            missing.discard(position)
            log.info("[Scanner] ✔ %s → '%s' camera", ip, position.upper())
        elif position:
            log.info("[Scanner] %s identifies as '%s' (already mapped).", ip, position)
        else:
            log.debug("[Scanner] %s is not an ESP32-CAM (no /id endpoint).", ip)

    if missing:
        log.warning("[Scanner] Still missing cameras after scan: %s", ", ".join(missing))

    return mappings
