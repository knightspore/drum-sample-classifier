# Sample Pack Generation with a Gan

## Generate Dataset Notebook

- Searches a designated path recursively for .wav files
- These files are not duplicated, but simply saved as a filename + path
- We use a (shitty) text search implementation to classify samples into classes
- This is exported as a .csv file

## Classification Notebook

- The model was created using [this tutorial](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

- There is a basic high-accuracy model (on seen, tested data) in `models/standard-10_8-512_16.pkl`. The naming convention listed is `readablename-epoch_batchsize...`

### Todo

- [X] Adjust `predict()` to accept single file paths
- [ ] Add K Fold Cross Validation

### Models

1. **standard-10_8-512_16.pkl**

This is a basic model trained with the NN from the article above and basic dataset. 80-90 percent accuracy on validation set data. 

2. ???

K Fold Cross Validation version of the above?