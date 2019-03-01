import math

def deg2num(zoom, lat_deg, lon_deg):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return zoom, xtile, ytile

# https://wiki.openstreetmap.org/wiki/Zoom_levels
# The sentinance paper uses a pixel=1m zoom level, but we shall go with something larger.

zoom = 14

# Use this website to get bounding boxes:
#               https://www.gps-coordinates.net/
# Western united states
# Top left Latitude: 48.995395 | Longitude: -124.866258
# Top right Latitude: 48.995395 | Longitude: -104.052404
# Bottom Right: Latitude: 31.74734 | Longitude: -104.052404
# Bottom Left: Latitude: 31.74734 | Longitude: -124.866258

# West US Data
zoom, startx, starty = deg2num(zoom, 48.995395, -124.866258)
zoom, endx, endy = deg2num(zoom, 31.74734, -104.052404)

for x in range(startx, endx + 1):
    for y in range(starty, endy + 1):
        # print(zoom, x, y)
        print(f"http://localhost/tile/{zoom}/{x}/{y}.png")
