docker-build:
	docker build -t felixlaumon/plankton .
	docker rmi -f `docker images --filter 'dangling=true' -q --no-trunc`

docker-bash:
	docker run -t -i --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm felixlaumon/plankton /bin/bash

.PHONY: docker-build docker-bash

