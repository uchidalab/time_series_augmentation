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


def dtw_graph2d(cost, DTW, path, prototype, sample):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
   # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    #cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[0]-0.5))

    #dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0]+1, path[1]+1, 'y')
    plt.xlim((-0.5, DTW.shape[0]-0.5))
    plt.ylim((-0.5, DTW.shape[0]-0.5))

    #prototype
    plt.subplot(2, 3, 4)
    plt.plot(prototype[:,0], prototype[:,1], 'b-o')

    #connection
    plt.subplot(2, 3, 5)
    for i in range(0,path[0].shape[0]):
        plt.plot([prototype[path[0][i],0], sample[path[1][i],0]],[prototype[path[0][i],1], sample[path[1][i],1]], 'y-')
    plt.plot(sample[:,0], sample[:,1], 'g-o')
    plt.plot(prototype[:,0], prototype[:,1], 'b-o')

    #sample
    plt.subplot(2, 3, 6)
    plt.plot(sample[:,0], sample[:,1], 'g-o')

    plt.tight_layout()
    plt.show()

def dtw_graph1d(cost, DTW, path, prototype, sample):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
   # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    p_steps = np.arange(prototype.shape[0])
    s_steps = np.arange(sample.shape[0])

    #cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[0]-0.5))

    #dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0]+1, path[1]+1, 'y')
    plt.xlim((-0.5, DTW.shape[0]-0.5))
    plt.ylim((-0.5, DTW.shape[0]-0.5))

    #prototype
    plt.subplot(2, 3, 4)
    plt.plot(p_steps, prototype[:,0], 'b-o')

    #connection
    plt.subplot(2, 3, 5)
    for i in range(0,path[0].shape[0]):
        plt.plot([path[0][i], path[1][i]],[prototype[path[0][i],0], sample[path[1][i],0]], 'y-')
    plt.plot(p_steps, sample[:,0], 'g-o')
    plt.plot(s_steps, prototype[:,0], 'b-o')

    #sample
    plt.subplot(2, 3, 6)
    plt.plot(s_steps, sample[:,0], 'g-o')

    plt.tight_layout()
    plt.show()