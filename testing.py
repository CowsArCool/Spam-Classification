import torch
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def computeBinaryAccuracy(output, trg):
    accuracy = torch.sum(
        torch.round(
            torch.sigmoid(
                output
            )
        ) == trg
    )/len(trg)

    return accuracy


def computeCrossEntropy(output, trg):
    predictions = torch.argmax(output, dim=1)

    accuracy = torch.sum(
        (predictions == trg)
    )/len(trg)

    return accuracy


def testModel(testloader,
              model,
              device,
              output_dim=1):

    if output_dim == None:
        raise ValueError(
            "output dim is missing you DUMB WHORE BITCASS HOE")

    elif output_dim == 1:
        binary = True

    else:
        binary = False

    precision, recall, accuracy = list(), list(), list()

    for batch in testloader:
        src, trg = batch['src'].to(
            device), batch['trg'].type(torch.float)

        src = src.transpose(0, 1)

        output = model(src).to('cpu').detach()

        if binary:
            accuracy.append(computeBinaryAccuracy(output, trg))

            output = torch.round(torch.sigmoid(output)).numpy()
            # print('Output:')
            # print(output)
            # print(trg)

            precision.append(precision_score(
                trg, output, average=None, zero_division=0))
            recall.append(recall_score(trg, output, average=None))

        if binary == False:
            accuracy.append(computeCrossEntropy(output, trg))

    if binary:
        print(
            f'\033[1m[Output] Binary Accuracy: {np.mean(accuracy)}\t\
                Recall: {np.mean(recall)}   Precision: {np.mean(precision)}\033[0m')

    if binary == False:
        print(f'\033[0mMultiClass Accuracy: {np.mean(accuracy)}\033[0m')
