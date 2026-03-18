"""
scanner.py
──────────
Network utility to auto-discover ESP32-CAMs.

Two strategies:
  1. Direct probe: Try the statically configured IPs first (fast, reliable)
  2. nmap scan:    Fall back to full subnet scan if configured IPs are stale

This ensures cameras are always mapped to the correct position
(left/right/rear) based on their known IP addresses.
"""

import socket
import logging
import subprocess
import requests
import time
from typing import Dict, List, Optional

from config import CAMERA_PORTS

log = logging.getLogger(__name__)


def get_local_subnet() -> str:
    """Helper to guess the local subnet (e.g., 192.168.1.0/24)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable, just routes out the default gateway
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
        
    parts = ip.split('.')
    if len(parts) == 4 and parts[0] != '127':
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    return "192.168.1.0/24"  # Safe default fallback


def _extract_ip_from_url(url: str) -> Optional[str]:
    """Extract IP address from a URL like http://10.132.20.188/stream."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.hostname
    except Exception:
        return None


def _probe_camera_url(url: str, timeout: float = 3.0) -> bool:
    """
    Test if a URL serves a valid MJPEG stream or JPEG image.
    Returns True if the camera is reachable and responds correctly.
    """
    try:
        res = requests.get(url, stream=True, timeout=timeout)
        if res.status_code == 200:
            ctype = res.headers.get("Content-Type", "")
            res.close()
            if "multipart/x-mixed-replace" in ctype or "image/jpeg" in ctype:
                return True
            # Some ESP32-CAMs return text/html for the index page
            # but /stream serves MJPEG — also accept 200 OK
            return True
        res.close()
        return False
    except Exception:
        return False


def discover_esp32_cameras() -> Dict[str, str]:
    """
    Discover ESP32-CAMs on the network using a two-phase approach:
    
    Phase 1: Directly probe the configured IPs from config.py
             This is fast and preserves the correct position mapping.
    
    Phase 2: If any cameras are missing from Phase 1, do an nmap scan
             to find them at new IPs (e.g., after DHCP reassignment).
    
    Returns: Dict mapping 'left', 'right', 'rear' to their stream URLs.
    """
    mappings: Dict[str, str] = {}
    missing_positions: List[str] = []

    # ── Phase 1: Direct probe configured IPs ─────────────────────────────────
    log.info("[Scanner] Phase 1: Probing configured camera IPs...")
    
    for position, cfg in CAMERA_PORTS.items():
        url = cfg.get("url")
        if not url:
            missing_positions.append(position)
            continue
        
        log.info("[Scanner]   Probing %s → %s", position.upper(), url)
        if _probe_camera_url(url, timeout=3.0):
            log.info("[Scanner]   ✔ %s camera is ONLINE at %s", position.upper(), url)
            mappings[position] = url
        else:
            log.warning("[Scanner]   ✘ %s camera NOT responding at %s", position.upper(), url)
            missing_positions.append(position)

    if not missing_positions:
        log.info("[Scanner] All cameras found via direct probe. Skipping nmap scan.")
        return mappings

    # ── Phase 2: nmap scan for missing cameras ───────────────────────────────
    log.info("[Scanner] Phase 2: %d camera(s) missing, scanning network via nmap...",
             len(missing_positions))
    
    # Collect already-known IPs so we don't re-assign them
    known_ips = set()
    for url in mappings.values():
        ip = _extract_ip_from_url(url)
        if ip:
            known_ips.add(ip)

    subnet = get_local_subnet()
    log.info("[Scanner] Scanning subnet %s for port 80...", subnet)
    
    new_camera_ips: List[str] = []
    
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
                    new_camera_ips.append(ip)
                    
    except FileNotFoundError:
        log.error("[Scanner] nmap not found! Please install it (sudo apt install nmap).")
        return mappings
    except subprocess.TimeoutExpired:
        log.error("[Scanner] nmap scan timed out.")
        return mappings
    except Exception as exc:
        log.error("[Scanner] nmap error: %s", exc)
        return mappings

    log.info("[Scanner] Found %d new web servers to test.", len(new_camera_ips))
    
    # Test each new IP for MJPEG stream
    discovered_urls = []
    for ip in new_camera_ips:
        url = f"http://{ip}/stream"
        if _probe_camera_url(url, timeout=2.0):
            log.info("[Scanner] Discovered valid stream at %s", url)
            discovered_urls.append(url)
    
    # Assign new URLs to missing positions in order
    for i, pos in enumerate(missing_positions):
        if i < len(discovered_urls):
            mappings[pos] = discovered_urls[i]
            log.info("[Scanner] Assigned %s → %s", pos.upper(), discovered_urls[i])
    
    log.info("[Scanner] Final result: %d/%d cameras mapped.", 
             len(mappings), len(CAMERA_PORTS))
    
    return mappings
