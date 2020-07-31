import numpy as np

def plot2d(x, y, x2=None, y2=None, x3=None, y3=None, xlim=(-1, 1), ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2)
    if x3 is not None and y3 is not None:
        plt.plot(x3, y3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, "")
    else:
        plt.show()
    return

def plot1d(x, x2=None, x3=None, ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    steps = np.arange(x.shape[0])
    plt.plot(steps, x)
    if x2 is not None:
        plt.plot(steps, x2)
    if x3 is not None:
        plt.plot(steps, x3)
    plt.xlim(0, x.shape[0])
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, "")
    else:
        plt.show()
    return