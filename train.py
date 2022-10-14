import argparse
from asyncio import MultiLoopChildWatcher
from operator import index
import os
import random
from xml.sax.xmlreader import InputSource
import tqdm
import torch
from sklearn.metrics import accuracy_score
import numpy as np

from eval_utils import downstream_validation
import utils
import data_utils
from data_utils import split_into_train_val
from model import Skipgram


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        input, target = self.dataset[index]
        return input, target

    def __len__(self):
        return len(self.dataset)

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== NOTE: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    # skipgram --> one word predicts multiple surrounding words
    input_outputs = []
    for sentence in encoded_sentences:
        for central_i in range(len(sentence)):
            # if there will not be any input-outputs from here on out (no matter what context window), then break
            if central_i > 5 and sentence[central_i-5] == 0:
                break

            # context half-size k (pi-k, pi-(k-1), ... pi, ... pi+(k-1), pi+k) is sampled between 2-6
            # this gives minimum 2 context words & up to 10 context words
            context_size = random.randint(2, 6)
            input = [sentence[central_i]]
            output = [-1]*10
            pointer = 0

            # should not create input-output pairs for sentence ends (i.e. seqs of 0 0 0 0 0 0... 0)
            # should create input-output pairs for context words that are within bounds
            for offset in range(1, context_size):
                position = central_i + offset
                if position < len(sentence) and sentence[position] != 0:
                    output[pointer] = sentence[position]
                    pointer += 1

                position = central_i - offset
                if position > 0  and sentence[position] != 0:
                    output[pointer] = sentence[position]
                    pointer += 1

            input_outputs.append((input, output))

    # splitting training and validation sets
    train, val = split_into_train_val(input_outputs)
    train_dataset = Custom_Dataset(train)
    val_dataset = Custom_Dataset(val)
    
    # creating data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)
    return train_loader, val_loader, index_to_vocab


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== NOTE: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = Skipgram(args.vocab_size, args.embedding_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== NOTE: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    learning_rate = 0.01

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return criterion, optimizer

def calculate_accuracy(predictions, labels):
    # record all intersection/union values
    ious = []
    skips = 0
    for i in range(len(predictions)):

        # find k labels that exist for word
        labelsThatExistForCentralWord = []
        for j in range(len(labels)):
            if labels[j][i] != -1:
                labelsThatExistForCentralWord.append(labels[j][i])

        labelsCount = len(labelsThatExistForCentralWord)
        if labelsCount == 0:
            skips += 1
            continue

        # for each output, find top k predicted words
        top_k = set()
        for j in range(labelsCount):
            index_of_max = torch.argmax(predictions[i])
            top_k.add(predictions[i][index_of_max])
            predictions[i][index_of_max] = 0

        # compare targets to predictions
        intersected = top_k.intersection(labelsThatExistForCentralWord)
        unioned = top_k.union(labelsThatExistForCentralWord)

        single_iou = len(intersected) / len(unioned)
        ious.append(single_iou)

    divisor = (len(predictions)-skips)
    if divisor != 0:
        return sum(ious)/divisor
    return 0

def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    batch_accuracy = []
    batch_sizes = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # # calculate the loss and train accuracy and perform backprop
        # # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs[0])

        # calculate prediction loss
        # pred_logits is 32x3000, multiHotLabels is 32x3000
        multiHotLabels = np.zeros((len(inputs[0]), args.vocab_size), dtype=np.int32)
        for oneLabelFromEachInput in range(len(labels)):
            for inputIdx in range(len(labels[oneLabelFromEachInput])):
                # only iterate through labels tensor while there are valid labels
                # note: label of -1 is included initially so that dynamic # of labels can be placed on tensor of uniform length
                label = labels[oneLabelFromEachInput][inputIdx]
                if label != -1:
                    multiHotLabels[inputIdx][label] = 1

        # creating tensor to be fed into loss function
        multiHotTensor = torch.from_numpy(multiHotLabels)

        loss = criterion(pred_logits, multiHotTensor.float())

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        # calculate accuracy per-batch, store batch-wise accuracy, compute average accuracy where current accuracy is
        batch_accuracy.append(calculate_accuracy(pred_logits, labels))
        batch_sizes.append(len(inputs[0]))

    # compute total accuracy based on weighted average of batch-wise accuracies
    achieved = 0
    possible = 0
    for i in range(len(batch_accuracy)):
        achieved += batch_accuracy[i]*batch_sizes[i]
        possible += batch_sizes[i]
    acc = achieved/possible
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=5, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=1,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== NOTE: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,
        help="size of embedding vector",
    )

    args = parser.parse_args()
    main(args)
