#!/usr/bin/env bash
# Pull docker image if necessary
if [ "$(docker image ls | grep 'willprice/nvidia-ffmpeg')" ]; then
    echo "willprice/nvidia-ffmpeg found!"
else
    echo "Pulling willprice/nvidia-ffmpeg..."
    docker pull willprice/nvidia-ffmpeg 
fi

# Run the docker
docker run \
    --rm -it --runtime=nvidia \
    --volume $PWD:/workspace \
    --volume "$(realpath -s ./data/EPIC_KITCHENS_2018)":/data/EPIC_KITCHENS_2018/ \
    --entrypoint bash \
    --user $(id -u):$(id -g) \
    willprice/nvidia-ffmpeg
