# Base Packages
import os
import numpy as np
import itertools
from numpy.core.fromnumeric import shape

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Random Packages
from alive_progress import alive_bar


def makedir_if_needed(directory):
    """Ensure directory if it doesn t exist .

    Args:
        directory ([path]): [path to create dir at]
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)


def trainModel(
    trainloader,
    testloader,
    testset,
    model,
    args,
    output_dim=1,
    wandb=None,
    device=torch.device('cpu')
):
    if wandb != None:
        enable_wandb = True

    else:
        enable_wandb = False

    ## Constants ##
    availabe_optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'SGD_nesterov': optim.SGD
    }

    if args.optimizer in availabe_optimizers:
        uninitialized_optim = availabe_optimizers.get(args.optimizer)
        if args.optimizer == 'Adam':
            optimizer = uninitialized_optim(model.parameters(),
                                            lr=args.learning_rate)

        elif args.optimizer == 'SGD':
            optimizer = uninitialized_optim(model.parameters(),
                                            lr=args.learning_rate,
                                            momentum=0)

        elif (args.optimizer == 'SGD_nesterov'):
            optimizer = uninitialized_optim(model.parameters(),
                                            lr=args.learning_rate,
                                            momentum=0.5,
                                            nesterov=True)

    if output_dim == 1:
        criterion = nn.BCEWithLogitsLoss()

    elif output_dim == None:
        raise ValueError("output_dim is none, thats stupid")

    else:
        criterion = nn.CrossEntropyLoss()

    running_dir = os.path.dirname(os.path.realpath(__file__))

    # Get directory for all saved models
    saved_models_dir = os.path.join(running_dir, 'saved_models')
    makedir_if_needed(saved_models_dir)

    # Get directory for this specific run
    save_dir = os.path.join(
        saved_models_dir, f'archatecture{model.name}lr{args.learning_rate}batch_size{args.batch_size}\
num_layers{args.num_layers}embedding_size{args.embedding_dim}optim{args.optimizer}')

    makedir_if_needed(save_dir)
    print(f'[Info] Model Directory: {save_dir}')

    # Runs until another file of the same name at the same location isnt found
    # it increases the count each time allowing for another model
    count = 0
    while (count != -1):
        count += 1
        save_path = os.path.join(save_dir, f'model_{count}')
        if (os.path.isfile(save_path) != True):
            count = -1

    wandb.watch(model)

    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_acc = []

        model.train()
        with alive_bar(len(trainloader),
                       title='Training', bar='smooth',
                       length=75) as bar:
            for batch_num, batch in enumerate(trainloader):
                src = (batch['src']
                       .to(device)
                       .transpose(0, 1)
                       )

                trg = batch['trg'].type(torch.float).to(device)

                optimizer.zero_grad()

                output = model(src)

                # assert output.shape[0] == trg.shape[0]

                loss = criterion(output, trg)
                epoch_losses.append(loss.item())

                if enable_wandb:
                    wandb.log(
                        {
                            'loss': loss,
                            'epoch': epoch
                        }
                    )

                try:
                    rounded_pred = torch.round(torch.sigmoid(output))  # WTF
                    correct = (rounded_pred == trg).float()

                    acc = correct.sum()/len(correct)
                    epoch_acc.append(acc)
                except:
                    epoch_acc = 0

                loss.backward()

                optimizer.step()

                bar.text(f'Epoch Step: {batch_num+1}')
                bar()

        model.eval()

        test_losses = []
        test_accuracies = []
        # with torch.nograd
        for batch in testloader:
            test_src = batch['src'].to(device)
            test_src = test_src.transpose(0, 1)

            test_trg = batch['trg'].type(torch.float).to(device)

            output = model(test_src)
            # print('Output Shape {}'.format(output.shape))
            # print('Target Shape: {}'.format(test_trg.shape))

            test_loss = criterion(output, test_trg)

            try:
                test_accuracy = (
                    torch.sum(
                        torch.round(
                            torch.sigmoid(output)
                        ) == test_trg)
                )/len(test_trg)

                test_losses.append(test_loss.item())
                test_accuracies.append(test_accuracy.item())

            except:
                test_accuracies = 0

            # print(output)

        table = wandb.Table(
            columns=['Text', 'Predicted Label', 'Target Label', 'Epoch'])

        text = []
        src_sentences = test_src.contiguous().transpose(0, 1)

        for sentence in src_sentences:
            text.append(
                " ".join(
                    [testset.itos[word.item()] for word in sentence]
                ).split('<pad>')[0]
            )

        # print(
        #     f'Len Sentence: {len(sentence)}, Output Shape: {output.shape}, test_trg shape {test_trg.shape}')

        for sentence, pred, label in zip(text, torch.sigmoid(output), test_trg):
            table.add_data(sentence, pred.item(), label.item(), epoch)
            # print(
            #     f'sentence: {sentence}, pred: {pred.item()}, label: {label.item()}')

        if enable_wandb:
            wandb.log(
                {
                    'test accuracy': np.mean(test_accuracies),
                    'test loss': np.mean(test_losses),
                    'output': wandb.Histogram(output.cpu().detach().numpy()),
                    'outputVector': output.cpu().detach().numpy(),
                    'predictions': table,
                    'epoch': epoch,
                }
            )

        print(
            f'\nEpoch: {epoch}\nAvg Loss: {np.round(np.mean(epoch_losses), decimals=3)}\
            \tTest Loss: {np.round(np.mean(test_losses), decimals=3)}\
            \tTest Accuracy: {np.round((np.mean(test_accuracies)*100), decimals=3)}%\
            ')

        if epoch % args.save_every == 0:
            checkpoint = {
                # saves all epochs in the same file with the epoch in their
                # indexable save name
                f'epoch:{epoch}_state_dict': model.state_dict(),
                f'epoch:{epoch}_optimizer': optimizer.state_dict()
            }

            # Save epoch to file
            torch.save(checkpoint, save_path)

    return model
