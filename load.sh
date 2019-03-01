#!/bin/bash
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker volume rm openstreetmap-data
docker volume create openstreetmap-data
docker run -v /your/dir/here/us-west-latest.osm.pbf:/data.osm.pbf -v openstreetmap-data:/var/lib/postgresql/10/main overv/openstreetmap-tile-server import
