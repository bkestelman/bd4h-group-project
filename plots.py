import os
import matplotlib.pyplot as plt
import config


def plot(title, x_label, y_label, series, save_name):

    plt.figure()
    plt.title(title)

    for s in series:
        plt.plot(s['x'], s['y'], label=s['label'], marker='.')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.savefig(os.path.join(config.plots_dir, save_name))
    # plt.show()


# train size vs AUC
series = [{
            'x': [3314, 5006, 6742, 8405, 10078, 11747, 13382, 15027, 16705, 18430],
            'y':  [0.818424353, 0.802821112, 0.787298265, 0.803622516, 0.79251624, 0.795748983, 0.785707912, 0.789085268, 0.781823877, 0.784813329],
            'label':'train'
        },
        {
            'x': [3314, 5006, 6742, 8405, 10078, 11747, 13382, 15027, 16705, 18430],
            'y':[0.694719955, 0.697640626, 0.709092853, 0.698554529, 0.717360193, 0.691415777, 0.699057712, 0.719967031, 0.713301436, 0.7224674],
            'label':'test'
        }]

plot(title='train size vs logistic regression AUC using BagOfWords', x_label='train size', y_label='logistic regression AUC', series=series, save_name='trainsize_vs_lrAUC.png')

# features vs AUC
series = [{
            'x': [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            'y':  [0.770067972, 0.787298265, 0.80487515, 0.8204873, 0.835062385, 0.848668732, 0.860354951, 0.872739076, 0.883781200],
            'label':'train'
        },
        {
            'x': [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            'y':[0.707187491, 0.709092853, 0.712517253, 0.709691091, 0.711592703, 0.707255003, 0.704535737,  0.700359318270471, 0.69848396],
            'label':'test'
        }]

plot(title='BagOfWords feature size vs logistic regression AUC', x_label='BagOfWords feature size', y_label='logistic regression AUC', series=series, save_name='bowfeaturesize_vs_lrAUC.png')
