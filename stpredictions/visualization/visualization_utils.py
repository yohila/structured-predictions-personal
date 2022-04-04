import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
# import torchvision
from typing import Optional, List


def plot_results(str_score: str, train_loss: List, valid_loss: List,
                 valid_score: List, train_score: Optional[List] = None) -> None:
    """
    Parameters
    ----------
    str_score: Title for second plot (ex: F1 Score, Hamming Loss, IOU)
    train_loss: List of train loss accross epochs
    valid_loss
    valid_score
    train_score
    """
    x = list(range(1, len(train_loss) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax2.set_title(str_score)
    ax2.set_ylabel(str_score)
    ax2.set_xlabel('epochs')

    ax1.plot(x, train_loss, label='loss_train')
    ax1.plot(x, valid_loss, label='loss_valid')

    ax2.plot(x, valid_score, label="validation")
    if train_score is not None:
        ax2.plot(x, train_score, label="train")

    ax2.legend()
    ax1.legend()
    plt.show()
