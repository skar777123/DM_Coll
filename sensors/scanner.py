"""
scanner.py
──────────
Network utility to auto-discover ESP32-CAMs.

Discovery strategies (in order of priority):
  1. mDNS:         Find cameras by hostname (blindspot-left.local, etc.)
  2. Direct probe: Try configured static IPs from config.py
  3. /id endpoint:  Query discovered cameras for their position identity
  4. nmap scan:     Full subnet scan as last resort

The /id endpoint on ESP32-CAMs returns JSON: {"position": "left", "ip": "...", "hostname": "..."}
This allows automatic position mapping even when IPs change.
"""

import socket
import logging
import subprocess
import requests
import time
from typing import Dict, List, Optional

from config import CAMERA_PORTS

log = logging.getLogger(__name__)

# mDNS hostnames for each camera position
MDNS_HOSTNAMES = {
    "left":  "blindspot-left.local",
    "right": "blindspot-right.local",
    "rear":  "blindspot-rear.local",
}


def get_local_subnet() -> str:
    """Helper to guess the local subnet (e.g., 192.168.1.0/24)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
        
    parts = ip.split('.')
    if len(parts) == 4 and parts[0] != '127':
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    return "192.168.1.0/24"


def _resolve_mdns(hostname: str) -> Optional[str]:
    """Resolve an mDNS hostname to an IP address."""
    try:
        ip = socket.gethostbyname(hostname)
        return ip
    except socket.gaierror:
        return None


def _probe_stream(url: str, timeout: float = 3.0) -> bool:
    """Test if a URL serves a valid MJPEG stream or JPEG image."""
    try:
        res = requests.get(url, stream=True, timeout=timeout)
        ok = res.status_code == 200
        res.close()
        return ok
    except Exception:
        return False


def _query_camera_id(ip: str, timeout: float = 2.0) -> Optional[str]:
    """
    Query the /id endpoint on an ESP32-CAM to get its position.
    Returns the position string ('left', 'right', 'rear') or None.
    """
    try:
        res = requests.get(f"http://{ip}/id", timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            position = data.get("position")
            if position in ("left", "right", "rear"):
                log.info("[Scanner] Camera at %s identifies as: %s", ip, position)
                return position
    except Exception:
        pass
    return None


def discover_esp32_cameras() -> Dict[str, str]:
    """
    Discover ESP32-CAMs on the network using a multi-phase approach:
    
    Phase 1: mDNS resolution (fastest, most reliable if ESP32 supports it)
    Phase 2: Direct probe configured IPs from config.py
    Phase 3: nmap scan + /id endpoint query for unknown cameras
    
    Returns: Dict mapping 'left', 'right', 'rear' to their stream URLs.
    """
    mappings: Dict[str, str] = {}
    all_positions = set(CAMERA_PORTS.keys())  # {'left', 'right', 'rear'}

    # ── Phase 1: mDNS Discovery ──────────────────────────────────────────────
    log.info("[Scanner] Phase 1: Trying mDNS resolution...")
    
    for position, hostname in MDNS_HOSTNAMES.items():
        ip = _resolve_mdns(hostname)
        if ip:
            url = f"http://{ip}/stream"
            if _probe_stream(url, timeout=2.0):
                log.info("[Scanner]   ✔ %s → %s (%s)", position.upper(), url, hostname)
                mappings[position] = url
            else:
                log.warning("[Scanner]   mDNS resolved %s → %s but stream not responding", hostname, ip)
        else:
            log.debug("[Scanner]   mDNS: %s not found", hostname)

    if len(mappings) == len(all_positions):
        log.info("[Scanner] All cameras found via mDNS!")
        return mappings

    missing = all_positions - set(mappings.keys())
    
    # ── Phase 2: Direct probe configured IPs ─────────────────────────────────
    log.info("[Scanner] Phase 2: Probing configured IPs for %d missing camera(s)...", len(missing))
    
    for position in list(missing):
        cfg = CAMERA_PORTS.get(position, {})
        url = cfg.get("url")
        if not url:
            continue
        
        log.info("[Scanner]   Probing %s → %s", position.upper(), url)
        if _probe_stream(url, timeout=3.0):
            log.info("[Scanner]   ✔ %s camera ONLINE at configured IP", position.upper())
            mappings[position] = url
            missing.discard(position)
        else:
            log.warning("[Scanner]   ✘ %s camera NOT at configured IP", position.upper())

    if not missing:
        log.info("[Scanner] All cameras found via direct probe.")
        return mappings

    # ── Phase 3: nmap scan + /id endpoint ────────────────────────────────────
    log.info("[Scanner] Phase 3: %d camera(s) still missing. Running nmap scan...", len(missing))
    
    known_ips = set()
    for url in mappings.values():
        try:
            from urllib.parse import urlparse
            known_ips.add(urlparse(url).hostname)
        except Exception:
            pass

    subnet = get_local_subnet()
    log.info("[Scanner] Scanning subnet %s for port 80...", subnet)
    
    candidate_ips: List[str] = []
    
    try:
        result = subprocess.check_output(
            ["nmap", "-p", "80", "--open", "-oG", "-", subnet], 
            stderr=subprocess.STDOUT, 
            timeout=30,
            text=True
        )
        
        for line in result.split("\n"):
            if "Host:" in line and "Ports: 80/open/tcp" in line:
                ip = line.split("Host: ")[1].split(" ")[0]
                if ip not in known_ips:
                    candidate_ips.append(ip)
                    
    except FileNotFoundError:
        log.error("[Scanner] nmap not found! Install with: sudo apt install nmap")
        return mappings
    except subprocess.TimeoutExpired:
        log.error("[Scanner] nmap scan timed out.")
        return mappings
    except Exception as exc:
        log.error("[Scanner] nmap error: %s", exc)
        return mappings

    log.info("[Scanner] Found %d candidate IPs to check.", len(candidate_ips))
    
    # Try /id endpoint on each candidate to identify its position
    unidentified_urls = []
    
    for ip in candidate_ips:
        stream_url = f"http://{ip}/stream"
        if not _probe_stream(stream_url, timeout=2.0):
            continue
            
        log.info("[Scanner] Stream found at %s — checking /id ...", ip)
        
        # Try the /id endpoint for smart identification
        position = _query_camera_id(ip, timeout=2.0)
        if position and position in missing:
            mappings[position] = stream_url
            missing.discard(position)
            log.info("[Scanner]   ✔ Identified as %s camera!", position.upper())
        elif position:
            log.info("[Scanner]   Camera identifies as %s but that position is already mapped.", position)
        else:
            # No /id endpoint — save for fallback assignment
            unidentified_urls.append(stream_url)
            log.info("[Scanner]   No /id endpoint — will assign by order")
    
    # Assign any remaining unidentified cameras to missing positions
    for pos in sorted(missing):
        if unidentified_urls:
            url = unidentified_urls.pop(0)
            mappings[pos] = url
            log.info("[Scanner] Assigned %s → %s (by order, no /id)", pos.upper(), url)
    
    log.info("[Scanner] Final result: %d/%d cameras mapped.", len(mappings), len(CAMERA_PORTS))
    
    if len(mappings) < len(all_positions):
        missing_str = ", ".join(all_positions - set(mappings.keys()))
        log.warning("[Scanner] Missing cameras: %s — check power and WiFi!", missing_str)
    
    return mappings
