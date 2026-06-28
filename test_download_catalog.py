import urllib.request
import json
import os

fab_url = "https://raw.githubusercontent.com/Stellarium/stellarium/v0.21.0/skycultures/western/constellationship.fab"
hip_url = "http://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I/239/hip_main&-out=HIP,RAICRS,DEICRS,Vmag&-out.max=unlimited&Vmag=<=5.0"

try:
    print("Downloading constellationship.fab...")
    with urllib.request.urlopen(fab_url, timeout=10) as response:
        fab_data = response.read().decode('utf-8')
    
    print("Downloading Hipparcos catalog...")
    with urllib.request.urlopen(hip_url, timeout=10) as response:
        hip_data = response.read().decode('utf-8')
        
    print("Processing Hipparcos catalog...")
    stars = {}
    lines = hip_data.split('\n')
    data_started = False
    for line in lines:
        if line.startswith('------') or (data_started and line.strip()):
            if not data_started:
                data_started = True
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    hip_id = int(parts[0].strip())
                    ra = float(parts[1].strip())
                    dec = float(parts[2].strip())
                    vmag = float(parts[3].strip())
                    stars[hip_id] = {
                        "ra": ra,
                        "dec": dec,
                        "vmag": vmag
                    }
                except ValueError:
                    continue
                    
    print(f"Loaded {len(stars)} stars.")
    
    print("Processing constellationship.fab...")
    constellations = {}
    fab_lines = fab_data.split('\n')
    for line in fab_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        abbr = parts[0]
        num_lines = int(parts[1])
        connections = []
        # The remaining parts are pairs of HIP IDs
        idx = 2
        for _ in range(num_lines):
            if idx + 1 < len(parts):
                try:
                    hip1 = int(parts[idx])
                    hip2 = int(parts[idx+1])
                    connections.append((hip1, hip2))
                except ValueError:
                    pass
                idx += 2
        if connections:
            constellations[abbr] = connections
            
    print(f"Loaded {len(constellations)} constellations.")
    
    # Save a merged, lightweight JSON
    catalog = {
        "stars": stars,
        "constellations": constellations
    }
    
    with open("c:\\ProgettiPy\\MilkyWayStacker\\core\\sky_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
        
    print("Successfully saved core/sky_catalog.json!")
    
except Exception as e:
    print(f"Error: {e}")
