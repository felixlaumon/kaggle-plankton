For http://www.kaggle.com/c/datasciencebowl

## Requirements

- Nvidia GPU
- cuDNN
- https://github.com/benanne/Lasagne
- https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne.py
- scikit-learn, scikit-image and other dependencies from Anaconda Python distribution

## TODO

- [ ] Pixel value jittering
- [ ] Try ADAM instead of RMSprop
- [ ] Try batch norm again??
- [ ] Try weight norm
- [ ] ipython upstart job doesn't work with DNN (probably because of path)
- [X] Confusion matrix?
- [X] Embarrassingly parallelize transform - isn't faster
- [X] Find out misclassified error across sub-categories to determine if necessary to utilize tree structure -> Implement expert system
- [X] Make sure test image averaging works
- [X] See if more neurons is actually necessary (check if the filters are dead or not) - Neuron not dead probably could use more layers or wider nets?
- [X] Use DNN instead of `cuda_convnet`
- [x] Benchmark real time augmentator and turn them into a single affine transform
- [x] Generate prediction based on transformed photos (uniformly averaged)
- [x] Mean subtraction augmentation
- [x] Mean subtractor for the test iterator
- [x] Write script to generate npy files
- [x] Write script to train model
- [x] Write script to write to submission files

## Reference

- http://benanne.github.io/2014/04/05/galaxy-zoo.html

## Docker

- Build image by
````
docker build -t felixlaumon/plankton .
````

- Attach to bash by
````
docker run -t -i --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm felixlaumon/plankton /bin/bash
````

- Clean up untagged build
````
docker rmi -f `docker images --filter 'dangling=true' -q --no-trunc`
````
