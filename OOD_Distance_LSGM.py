import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn import metrics
from sklearn.mixture import BayesianGaussianMixture
from torchvision import datasets, transforms

import models

parser = argparse.ArgumentParser(description='PyTorch code: FGSM detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--net_type', default='resnet', help='resnet | densenet')
parser.add_argument('--n_components', type=int, default=50, help='components # of Gaussian Mixture')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()
print(args)


def main():
    # set parameters
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'

    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load model
    if args.net_type == 'resnet':
        model = models.ResNet34(num_c=num_classes)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    else:
        model = models.DenseNet3(depth=100, num_classes=num_classes)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)), ])

    model.load_state_dict(torch.load(pre_trained_net))
    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=os.path.join(args.dataroot, args.dataset + '-data'), train=True, download=True,
                transform=test_transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=os.path.join(args.dataroot, args.dataset + '-data'), train=False, download=True,
                transform=test_transform),
            batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=os.path.join(args.dataroot, args.dataset + '-data'), train=True, download=True,
                transform=test_transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=os.path.join(args.dataroot, args.dataset + '-data'), train=False, download=True,
                transform=test_transform),
            batch_size=args.batch_size, shuffle=False)

    print('load dataset: ' + args.dataset)

    svhn_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root=os.path.join(args.dataroot, 'svhn-data'), split='test', download=True,
            transform=test_transform,
        ),
        batch_size=args.batch_size, shuffle=False)

    imagenet_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            os.path.expanduser(os.path.join(args.dataroot, 'Imagenet_resize')),
            transform=test_transform
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    lsun_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            os.path.expanduser(os.path.join(args.dataroot, 'LSUN_resize')),
            transform=test_transform
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # generate features
    pool_sizes = [32, 32, 16, 8, 4]
    train_list = []
    model.eval()
    with torch.no_grad():
        # extract training data
        for itr, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = model.feature_list(input)

            # process the data
            for i, layer in enumerate(out_list):
                assert layer.dim() == 4
                if itr == 0:
                    print(tuple(layer.shape), '->', end=' ')
                layer = F.avg_pool2d(layer, pool_sizes[i])
                out_list[i] = layer.reshape(layer.shape[0], -1)
                if itr == 0:
                    print(tuple(out_list[i].shape))

            # save data to list
            train_list.append([layer.cpu() for layer in out_list] + [y.cpu(), target])
            if itr % 50 == 49:
                print((itr + 1) * args.batch_size)

    train_features = [np.concatenate(f) for f in zip(*train_list)]
    n_layers = len(train_features) - 2
    correct_train = np.argmax(train_features[n_layers], axis=1) == train_features[n_layers + 1]
    print('correct number:', np.sum(correct_train))

    test_list = []
    model.eval()
    with torch.no_grad():
        for itr, (input, target) in enumerate(test_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = model.feature_list(input)
            for i, layer in enumerate(out_list):
                assert layer.dim() == 4
                layer = F.avg_pool2d(layer, pool_sizes[i])
                layer = layer.reshape(layer.shape[0], -1).cpu().numpy()
                out_list[i] = layer
            test_list.append(out_list + [y.cpu(), target])
            if itr % 50 == 49:
                print((itr + 1) * args.batch_size)

        for itr, (input, target) in enumerate(svhn_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = model.feature_list(input)
            for i, layer in enumerate(out_list):
                layer = F.avg_pool2d(layer, pool_sizes[i])
                layer = layer.reshape(layer.shape[0], -1).cpu().numpy()
                out_list[i] = layer
            test_list.append(out_list + [y.cpu(), target])
            if itr % 50 == 49:
                print((itr + 1) * args.batch_size)
                break

        for itr, (input, target) in enumerate(imagenet_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = model.feature_list(input)
            for i, layer in enumerate(out_list):
                layer = F.avg_pool2d(layer, pool_sizes[i])
                layer = layer.reshape(layer.shape[0], -1).cpu().numpy()
                out_list[i] = layer
            test_list.append(out_list + [y.cpu(), target])
            if itr % 50 == 49:
                print((itr + 1) * args.batch_size)

        for itr, (input, target) in enumerate(lsun_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = model.feature_list(input)
            for i, layer in enumerate(out_list):
                layer = F.avg_pool2d(layer, pool_sizes[i])
                layer = layer.reshape(layer.shape[0], -1).cpu().numpy()
                out_list[i] = layer
            test_list.append(out_list + [y.cpu(), target])
            if itr % 50 == 49:
                print((itr + 1) * args.batch_size)

    test_features = [np.concatenate(f) for f in zip(*test_list)]
    correct_test = np.argmax(test_features[n_layers][:10000], axis=1) == test_features[n_layers + 1][:10000]
    print('correct number:', np.sum(correct_test))

    # train clustering
    labels_train, labels_test = [], []
    probs_train, probs_test = [], []
    gmm_list = []
    # for layers, train...
    for i, features in enumerate(train_features):
        print('layer', i, ':', features.shape)
        if i == n_layers:  # last layer
            break

        x_train = train_features[i]
        x_test = test_features[i]
        # scaler = StandardScaler().fit(x_train) x_train = scaler.transform(x_train) x_test = scaler.transform(x_test)
        # train kmeans gmm = GaussianMixture(n_components=component_list[i], covariance_type='diag', max_iter=1000,
        # init_params='kmeans', reg_covar=1e-6, verbose=1, random_state=seed)
        gmm = BayesianGaussianMixture(n_components=args.n_components, covariance_type='diag', max_iter=2000,
                                      weight_concentration_prior_type='dirichlet_process',
                                      weight_concentration_prior=0.1, reg_covar=1e-6, init_params='kmeans')
        gmm.fit(x_train)
        gmm_list.append(gmm)
        #     gmm = gmm_list[i]

        # determine predicted labels
        # import sklearn.preprocessing
        # x_train = sklearn.preprocessing.normalize(x_train)
        # x_test = sklearn.preprocessing.normalize(x_test)
        x_train_labels = gmm.predict(x_train)
        x_test_labels = gmm.predict(x_test)
        labels_train.append(x_train_labels)
        labels_test.append(x_test_labels)

        # save the probilities
        x_train_probs = gmm.score_samples(x_test)
        x_test_probs = gmm._estimate_weighted_log_prob(x_test)
        probs_train.append(x_train_probs)
        probs_test.append(x_test_probs)
        #     print(x_test_probs)

        print('     \tcifar10 \tsvhn    \ttinyimagenet \tlsun')
        print('avg: \t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
            x_test_probs[:10000].mean(), x_test_probs[10000:20000].mean(), x_test_probs[20000:30000].mean(),
            x_test_probs[30000:40000].mean()))
        print('std: \t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
            x_test_probs[:10000].std(), x_test_probs[10000:20000].std(), x_test_probs[20000:30000].std(),
            x_test_probs[30000:40000].std()))
        print('max: \t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
            x_test_probs[:10000].max(), x_test_probs[10000:20000].max(), x_test_probs[20000:30000].max(),
            x_test_probs[30000:40000].max()))
        print('min: \t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
            x_test_probs[:10000].min(), x_test_probs[10000:20000].min(), x_test_probs[20000:30000].min(),
            x_test_probs[30000:40000].min()))
        print()

    path_train = np.vstack(labels_train).T
    path_test = np.vstack(labels_test).T
    y_test = np.concatenate((np.zeros(10000), np.ones(30000)))
    # incorrect as ood
    # y_test[:10000][~correct_test] = 1

    bigram = []
    for i in range(0, n_layers - 1):
        count = np.zeros([args.n_components, args.n_components]) + 1e-8
        for j in range(50000):
            path = path_train[j]
            u, v = path[i], path[i + 1]
            count[u][v] += 1

        for j in range(args.n_components):
            count[j] /= count[j].sum()
        count = np.log(count)

        bigram.append(count)

    def lsgm_score(j):
        m = probs_test[1][j].reshape(-1, 1) + probs_test[2][j].reshape(1, -1)  # 对应相加
        # m.shape == (k1, k2)
        m += bigram[1]  # 乘转移概率

        # layer 2->3
        for i in range(3, n_layers - 1):
            m = logsumexp(m, axis=0)
            m = m[:, np.newaxis] + probs_test[i][j].reshape(1, -1)
            m += bigram[i - 1]  # layer i-1 -> i

        return logsumexp(m)

    scores = np.array([lsgm_score(i) for i in range(40000)])
    scores = -scores
    print(scores)

    print('svhn')
    test(y_test[:20000], scores[:20000])
    print('tinyimagenet')
    test(np.concatenate((y_test[:10000], y_test[20000:30000])), np.concatenate((scores[:10000], scores[20000:30000])))
    print('lsun')
    test(np.concatenate((y_test[:10000], y_test[30000:40000])), np.concatenate((scores[:10000], scores[30000:40000])))


def test(y_true, scores):
    # roc
    #     print(y_true.shape, scores.shape)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    roc_auc = metrics.auc(fpr, tpr)
    # tnr
    tpr95 = np.where(tpr >= 0.95)[0][0]
    tnr = 1 - fpr

    print('TNR at TPR 95%: {:.4f}'.format(tnr[tpr95]))
    print('AUROC         : {:.4f}'.format(roc_auc))


if __name__ == '__main__':
    main()
