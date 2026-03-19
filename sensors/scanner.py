"""
scanner.py
──────────
Network utility to auto-discover ESP32-CAMs.

Logic:
1. Fetch IP of the Raspberry Pi.
2. Exclude the IP of the gateway, Raspberry Pi IP, and network IP.
3. Detect the IP of the range in which the Raspberry IP is connected.
4. Using nmap, fetch the IP of the ESP32 cam.
5. Assign it using the /id endpoint to fetch the camera visuals.
"""

import os
import socket
import logging
import subprocess
from typing import Optional, Iterable, Set, Dict, Tuple
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


def _get_expected_positions() -> Set[str]:
    try:
        from config import CAMERA_PORTS  # type: ignore
        if isinstance(CAMERA_PORTS, dict) and CAMERA_PORTS:
            return {str(k) for k in CAMERA_PORTS.keys()}
    except Exception:
        pass
    return {"left", "right", "rear"}


def _get_candidate_http_ports() -> Set[int]:
    """
    Derive ports to scan from config CAMERA_PORTS URLs when available.
    Defaults to {80} which is typical for ESP32-CAM sketches.
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
    """Fetch the IP of the Raspberry Pi / Host Machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def get_gateway_ip() -> str:
    """Fetch the IP of the Default Gateway to exclude it."""
    try:
        if os.name == 'nt':
            # Windows fallback
            out = subprocess.check_output(["route", "print", "0.0.0.0"], text=True)
            for line in out.split('\n'):
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "0.0.0.0":
                    return parts[2]
        else:
            # Linux (Raspberry Pi)
            with open("/proc/net/route") as fh:
                for line in fh:
                    fields = line.strip().split()
                    if len(fields) > 3 and fields[1] == '00000000' and int(fields[3], 16) & 2:
                        import struct
                        return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except Exception as e:
        log.warning(f"Failed to get gateway IP: {e}")
    return ""


def _query_camera_id(ip: str, timeout: float = 3.0) -> Optional[str]:
    """
    Query the /id endpoint on the ESP32-CAM to get its assigned visual position.
    Returns the position string ('left', 'right', 'rear') or None.
    """
    res = None
    try:
        # Use a connect/read timeout pair for better behavior on flaky WiFi.
        res = requests.get(f"http://{ip}/id", timeout=(timeout, timeout))
        if res.status_code == 200:
            data = res.json()
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


def discover_esp32_cameras() -> dict:
    """
    Implements the requested logic:
    1. Fetch Pi IP
    2. Exclude Gateway, Pi IP, Network IP
    3. Nmap scan the subnet range
    4. Fetch /id to assign visuals
    """
    mappings: Dict[str, str] = {}
    
    local_ip = get_local_ip()
    gateway_ip = get_gateway_ip()
    
    if local_ip == '127.0.0.1':
        log.error("[Scanner] Could not determine local IP. Network offline.")
        return mappings
        
    base_ip = local_ip.rsplit('.', 1)[0]
    network_ip = f"{base_ip}.0"
    broadcast_ip = f"{base_ip}.255"
    target_range = f"{base_ip}.0/24"

    expected_positions = _get_expected_positions()
    scan_ports = sorted(_get_candidate_http_ports())
    scan_ports_arg = ",".join(str(p) for p in scan_ports)
    
    # ── Exclude IP of gateway, Raspberry Pi IP, and network IP ───────────────
    excludes = [local_ip, network_ip, broadcast_ip]
    if gateway_ip:
        excludes.append(gateway_ip)
        
    exclude_str = ",".join(excludes)
    
    log.info("[Scanner] Host IP: %s | Gateway: %s", local_ip, gateway_ip)
    log.info("[Scanner] Scanning range: %s ports=%s (Excluding: %s)", target_range, scan_ports_arg, exclude_str)
    
    candidate_ips = []
    try:
        # ── Using nmap to fetch IPs ───────────────────────────────────────────
        result = subprocess.check_output(
            ["nmap", "-p", scan_ports_arg, "--open", "--exclude", exclude_str, "-oG", "-", target_range],
            stderr=subprocess.STDOUT, 
            timeout=45,
            text=True
        )
        for line in result.split("\n"):
            # Grepable output example:
            # Host: 192.168.1.10 ()  Ports: 80/open/tcp//http///, 443/closed/tcp//https///
            if "Host:" in line and "Ports:" in line and "/open/tcp" in line:
                ip = line.split("Host: ")[1].split(" ")[0]
                candidate_ips.append(ip)
                
    except FileNotFoundError:
        log.error("[Scanner] nmap not found! Install it (Linux: sudo apt install nmap, Windows: install Nmap).")
        return mappings
    except subprocess.TimeoutExpired:
        log.error("[Scanner] nmap scan timed out.")
        return mappings
    except Exception as exc:
        log.error("[Scanner] nmap error: %s", exc)
        return mappings

    log.info("[Scanner] Found %d candidate web servers in range: %s", len(candidate_ips), candidate_ips)
    
    # ── Assign it from where it will get the visuals (via /id) ────────────────
    missing = set(expected_positions)
    
    for ip in candidate_ips:
        if not missing:
            break
            
        stream_url = f"http://{ip}/stream"
        
        position = _query_camera_id(ip, timeout=3.0)
        
        if position and position in missing:
            mappings[position] = stream_url
            missing.remove(position)
            log.info("[Scanner]   ✔ IP %s identifies as '%s' camera! Assigned visuals.", ip, position.upper())
        elif position:
            log.info("[Scanner]   IP %s identifies as '%s' but already mapped.", ip, position)
        else:
            log.debug("[Scanner]   IP %s is not an ESP32 camera (no /id endpoint).", ip)
            
    if missing:
        log.warning("[Scanner] Missing cameras after scan: %s", ", ".join(missing))
    
    return mappings

