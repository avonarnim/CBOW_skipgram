# CSCI 499 HW 2: Word Embeddings

## Implementation Choices

### Model

For this assignment, I implemented skipgram. In `model.py`, you can find my model structure, which includes an embedding layer, a fully-connected layer, and a softmax layer. The embedding layer is initialized with uniform values of `1/vocab_size`. For an input of `batch_size` entries, this generates `batch_size*vocab_size` outputs, with each output value in a range from [0.0, 1.0].

### Dataloader

To create an initial input-output set, I looped through all words in all encoded sentences and then generated a list of context words for each central word. The outputs were formatted in a list of length 10, where values above -1 indicated word_to_index outputs and values of -1 indicated that there were less than 10 context words. During training, these lists are converted into multi-hot tensors with values of 1 at each word_to_index output and 0 elsewhere.

After creating an input-output set, the data is split into training and validation sets and then fed into dataloaders. Shuffling only happens after the train-validation split so that the validation input-outputs are likely from a different set of books than the training set.

One extra step taken in the data loading process is to skip all potential input-output pairs that will not have any labels (i.e. when a central word with context_size=10 will only see padding in the rest of a sentence). This helps speed up training.

### Training

The model uses BCEWithLogitsLoss as its loss function and SGD as its optimizer. A relatively high learning rate (0.01) was used so that fewer epochs could be run with still a high rate of parameter tuning.

The training process includes obtaining a batch, creating multiHot vectors for each input word's set of labels, and performing prediction/loss/optimization steps.

## Performance

After 5 epochs, my model had an MRR of 0.0022 and MR of 457 for semantic relations and an MRR of 0.0010 and MR of 1048 for syntactic relations.
The loss across epochs for both training and validation sets remained at 0.69335, which made me suspect that the model optimization step might not be working properly, but the embeddings outputted by the model were different from the initial weightings, which indicates some update did happen. The accuracy remained at 0 across training and validation datasets.
My embedding dimension count was 128, which I picked in spite of the Word2Vec paper suggesting that a higher dimension count (300-600) would be better; I made this decision because I assume a lower dimension count will speed up execution slightly and also because the vocbulary size we are dealing with (3000) is much less than the vocabulary size in Word2Vec. This distinction made me think a slight simplification of the model would not harm accuracy attainable for the given dataset and vocab size.
I used only 5 epochs, training with a 0.01 learning rate, performing evaluations at every epoch. This was done because I did not have the time to make sure everything ran for 30 epochs straight (which would take me ~15 hours).

## Bonus

### Varying context window size

I completed the bonus task of creating a varying context size that was sampled from the range of [2, 10]. This was done, as mentioned earlier, by creating output lists of the maximum possible size (10) and pre-filling them with -1 (which I knew no words were encoded as). Therefore, when making multiHot vectors at train time, I would only use positive values from the label sets. The idea behind creating varying context sizes is to take into account that the closest words to a central word are probably more commonly correlated than distant words, but that distant words may also have some correlation.

### Skipgram

I implemented skipgram (as far as I know how to). This included some time being spent on creating a custom accuracy calculation function. The accuracy function is fed a matrix of size (batch_size)x(vocab_size) as well as a list of labels for the batch entries. The accuracy function will then gather all labels (L) for a given batch entry and compare them to the top |L| logits predicted by the model in an intersection-over-union calculation.

This calculation is done for every batch and then the average over the epoch is calculated.

## Analysis of Released Code

The in vitro task involved in this code is predicting context words for each word in a corpus derived from classic books such as Walden, The Picture of Dorian Gray, and The Iliad. The in vivo task being evaluated is accuracy at predicting word-pair relations (analogies). For example, the model is tested whether given words such as `fat`, `thin`, and `back` in the pretext of guessing antonyms, it can predict `front`. The metrics being used for the analogies dataset are Mean Reciprocal Rank (MRR) and Mean Rank. MRR indicates what ranking a guess is placed at. For example, if for the previous example, the model predicted `front` as the

- 1st most-likely word --> MRR = 1
- 2nd most-likely word --> MRR = 1/2
- 3rd most-likely word --> MRR = 1/3
- 500th most-likely word --> MRR = 0.002
- 3000th most-likely word --> MRR = 0.00033.
  MR is the reciprocal of this:
- 1st most-likely word --> MR = 1
- 2nd most-likely word --> MR = 2
- 3rd most-likely word --> MR = 3
- 500th most-likely word --> MR = 500
- 3000th most-likely word --> MR = 3000.
  One aspect of these metrics that can underevaluate model performance is the fact that certain semantic categories like
  "capitals" will undoubtely have very low model performance because the texts we assess are generally non-information
  based and therefore will not have geographic information presented very often. This - when factored into the overall
  semantic accuracy - lowers the score beyond what is (arguably) fair.

## Fixes

Note: after reviewing the completed homework, I realized that I was inappropriately applying a softmax on the outputs of the
model's FC layer, which in turn causes the loss function (which expects logits and performs its own sigmoid operation) to
be operating on top of probability distributions. This (at the minimum) causes learning to be extremely slow.

After getting rid of the softmax, the model began learning better. After 5 epochs, the MRR for semantic relations increased
from 0.0022 to 0.0028 and the MR shrunk from 457 to 356. The MRR for syntactic relations increased from 0.0010 to 0.0024 and
the MR shrunk from 1048 to 425.

Further, the training loss shrunk to 0.2537, and the validation loss shrunk to 0.0514.
