"""
scanner.py
──────────
Network utility to auto-discover ESP32-CAMs using nmap.
Finds devices serving HTTP on port 80 and checks if they serve an MJPEG stream.
"""

import socket
import logging
import subprocess
import requests
import time
from typing import Dict, List

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
    return "192.168.1.0/24" # Safe default fallback

def discover_esp32_cameras() -> Dict[str, str]:
    """
    Scans the local network for port 80 using nmap, tests matching IPs 
    for an MJPEG stream on '/stream', and maps them to positions.
    Returns: Dict mapping 'left', 'right', 'rear' to their new stream URLs.
    """
    subnet = get_local_subnet()
    log.info("[Scanner] Starting nmap scan on subnet %s for port 80...", subnet)
    
    active_ips: List[str] = []
    
    try:
        # We use nmap to find all IPs with open port 80 rapidly.
        # This requires nmap installed on the Pi: `sudo apt install nmap`
        result = subprocess.check_output(
            ["nmap", "-p", "80", "--open", "-oG", "-", subnet], 
            stderr=subprocess.STDOUT, 
            timeout=30,
            text=True
        )
        
        # Parse the greppable output (-oG) to find active IPs
        for line in result.split("\n"):
            if "Host:" in line and "Ports: 80/open/tcp" in line:
                ip = line.split("Host: ")[1].split(" ")[0]
                active_ips.append(ip)
                
    except FileNotFoundError:
        log.error("[Scanner] nmap not found! Please install it (sudo apt install nmap).")
        return {}
    except subprocess.TimeoutExpired:
        log.error("[Scanner] nmap scan timed out.")
        return {}
    except Exception as exc:
        log.error("[Scanner] nmap error: %s", exc)
        return {}

    log.info("[Scanner] nmap found %d potential web servers on port 80. Testing for streams...", len(active_ips))
    
    discovered_urls = []
    
    # Test each IP to see if it responds to /stream with MJPEG
    for ip in active_ips:
        url = f"http://{ip}/stream"
        try:
            # Send a HEAD or quick GET request (stream=True and close immediately)
            res = requests.get(url, stream=True, timeout=2.0)
            if res.status_code == 200:
                ctype = res.headers.get("Content-Type", "")
                if "multipart/x-mixed-replace" in ctype or "image/jpeg" in ctype:
                    log.info("[Scanner] Discovered valid stream at %s", url)
                    discovered_urls.append(url)
            res.close()
        except Exception:
            pass # Not a camera or unreachable port 80 daemon
            
    log.info("[Scanner] Total valid camera streams found: %d", len(discovered_urls))
    
    # Map URLs to positions
    # If the ESP32-CAMs don't have distinct ways to identify themselves (like specific headers),
    # we assign them based on order or fallback.
    # Ideally, they'd have custom headers or distinct URIs, but for now we distribute them:
    positions = ["left", "right", "rear"]
    mappings = {}
    
    for i, url in enumerate(discovered_urls):
        if i < len(positions):
            pos = positions[i]
            mappings[pos] = url
            
    return mappings
