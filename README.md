For http://www.kaggle.com/c/datasciencebowl

## TODO

- [ ] Try weight norm
- [ ] See if more neurons is actually necessary (check if the filters are dead or not)
- [ ] Find out misclassified error across sub-categories to determine if necessary to utilize tree structure
- [ ] Try batch norm again??
- [ ] Pixel value jittering
- [ ] Embarrassingly parallelize transform?
- [X] Make sure test image averaging works
- [X] Use DNN instead of cuda_convnet
- [x] Benchmark real time augmentator and turn them into a single affine transform
- [x] Generate prediction based on transformed photos (uniformly averaged)
- [x] Mean subtraction augmentation
- [x] Mean subtractor for the test iterator
- [x] Write script to generate npy files
- [x] Write script to train model
- [x] Write script to write to submission files

## Reference

- http://benanne.github.io/2014/04/05/galaxy-zoo.html
