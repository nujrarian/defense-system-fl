import mxnet as mx
import numpy as np
from copy import deepcopy
import time
from numpy import random
from mxnet import nd, autograd, gluon
from sklearn.metrics import roc_auc_score
import argparse
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
import csv

def simple_mean(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1, keepdims=1)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * mean_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return mean_nd, distance


# trimmed mean
def trim(old_gradients, param_list, net, lr, b=0, hvp=None):
    '''
    gradients: the list of gradients computed by the worker devices
    net: the global model
    lr: learning rate
    byz: attack
    f: number of compromised worker devices
    b: trim parameter
    '''
    if hvp is not None:
        pred_grad = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    # sort
    sorted_array = nd.array(np.sort(nd.concat(*param_list, dim=1).asnumpy(), axis=-1), ctx=mx.gpu(5))
    # trim
    n = len(param_list)
    m = n - b * 2
    trim_nd = nd.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=1)

    # update global model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size

    return trim_nd, distance


def median(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    if len(param_list) % 2 == 1:
        median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2]
    else:
        median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2: len(param_list) // 2 + 1].mean(axis=-1, keepdims=1)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * median_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return median_nd, distance


def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()

def nearest_distance(gradient, c_p):
    sorted_distance = nd.square(c_p - gradient).sum(axis=1).sort(axis=0)
    return sorted_distance[1].asscalar()

def krum(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    num_params = len(param_list)
    q = b
    if num_params <= 2:
        # if there are too few clients, randomly pick one as Krum aggregation result
        random_idx = np.random.choice(num_params)
        krum_nd = nd.reshape(param_list[random_idx], shape=(-1, 1))
    else:
        if num_params - b - 2 <= 0:
            q = num_params-3
        v = nd.concat(*param_list, dim=1)
        scores = nd.array([score(gradient, v, q) for gradient in param_list])
        min_idx = int(scores.argmin(axis=0).asscalar())
        krum_nd = nd.reshape(param_list[min_idx], shape=(-1, 1))

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * krum_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return krum_nd, distance

def no_byz(v, f):
    return v


def partial_trim(v, f):
    '''
    Partial-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    # first compute the statistics
    vi_shape = v[0].shape
    all_grads = nd.concat(*v, dim=1)
    adv_grads = all_grads[:, :f]
    e_mu = nd.mean(adv_grads, axis=1)  # mean
    e_sigma = nd.sqrt(nd.sum(nd.square(nd.subtract(adv_grads, e_mu.reshape(-1, 1))), axis=1) / f)  # standard deviation

    for i in range(f):
        # apply attack to compromised worker devices with randomness
        v[i] = (e_mu - nd.multiply(e_sigma, nd.sign(e_mu)) * 3.5).reshape(vi_shape)

    return v


def full_trim(v, f):
    '''
    Full-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    # first compute the statistics
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    for i in range(f):
        # apply attack to compromised worker devices with randomness
        random_12 = 2
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v

def score(gradient, v, f):
    num_neighbours = int(v.shape[1] - 2 - f)
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()


def krum(v, f):
    if len(v) - f - 2 <= 0:
        f = len(v) - 3
    if len(v[0].shape) > 1:
        v_tran = nd.concat(*v, dim=1)
    else:
        v_tran = v
    scores = nd.array([score(gradient, v_tran, f) for gradient in v])
    min_idx = int(scores.argmin(axis=0).asscalar())
    krum_nd = nd.reshape(v[min_idx], shape=(-1,))
    return min_idx, krum_nd



def scaling_attack(v, f, epsilon=0.01):
    scaling_factor = len(v)
    for param_id in range(f):
        v[param_id] = v[param_id]*scaling_factor
    return v


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
np.warnings.filterwarnings('ignore')



def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.cpu())  #ctx=mx.gpu(args.gpu) if using gpu
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def params_convert(net):
    tmp = []
    for param in net.collect_params().values():
        tmp.append(param.data().copy())
    params = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
    return params


def clip(a, b, c):
    tmp = nd.minimum(nd.maximum(a, b), c)
    return tmp


def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    print(silhouette_score(score.reshape(-1, 1), label_pred))


def main(args):
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    with ctx:

        batch_size = args.batch_size

        if args.dataset == 'mnist':
            num_inputs = 28 * 28
            num_outputs = 10
            if args.net == 'mlr':
                input_size = (1, 28 * 28)
            else:
                input_size = (1, 1, 28, 28)
        else:
            raise NotImplementedError

        # Multiclass Logistic Regression
        MLR = gluon.nn.Sequential()
        with MLR.name_scope():
            MLR.add(gluon.nn.Dense(num_outputs))

        # two-layer fully connected nn
        fcnn = gluon.nn.Sequential()
        with fcnn.name_scope():
            fcnn.add(gluon.nn.Dense(256, activation="relu"))
            fcnn.add(gluon.nn.Dense(256, activation="relu"))
            fcnn.add(gluon.nn.Dense(num_outputs))

        # CNN
        cnn = gluon.nn.Sequential()
        with cnn.name_scope():
            cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # The Flatten layer collapses all axis, except the first one, into one axis.
            cnn.add(gluon.nn.Flatten())
            cnn.add(gluon.nn.Dense(512, activation="relu"))
            cnn.add(gluon.nn.Dense(num_outputs))

        def evaluate_accuracy(data_iterator, net):

            acc = mx.metric.Accuracy()
            for i, (data, label) in enumerate(data_iterator):
                if args.net == 'mlr':
                    data = data.as_in_context(ctx).reshape((-1, num_inputs))
                    label = label.as_in_context(ctx)
                else:
                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx)
                output = net(data)
                predictions = nd.argmax(output, axis=1)
                acc.update(preds=predictions, labels=label)

            return acc.get()[1]


        def evaluate_backdoor(data_iterator, net):
            target = 0
            acc = mx.metric.Accuracy()
            for i, (data, label) in enumerate(data_iterator):
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                data[:, :, 26, 26] = 1
                data[:, :, 26, 24] = 1
                data[:, :, 25, 25] = 1
                data[:, :, 24, 26] = 1
                remaining_idx = list(range(data.shape[0]))
                for example_id in range(data.shape[0]):
                    if label[example_id] != target:
                        label[example_id] = target
                    else:
                        remaining_idx.remove(example_id)

                output = net(data)
                predictions = nd.argmax(output, axis=1)
                predictions = predictions[remaining_idx]
                label = label[remaining_idx]
                acc.update(preds=predictions, labels=label)

            return acc.get()[1]

        def evaluate_edge_backdoor(data, net):
            acc = mx.metric.Accuracy()
            output = net(data)
            label = nd.ones(len(data)).as_in_context(ctx)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
            return acc.get()[1]

        # decide attack type
        if args.byz_type == 'partial_trim':
            # partial knowledge trim attack
            byz = partial_trim
        elif args.byz_type == 'full_trim':
            # full knowledge trim attack
            byz = full_trim
        elif args.byz_type == 'no':
            byz = no_byz
        elif args.byz_type == 'backdoor' or 'dba' or 'edge':
            byz = scaling_attack
        elif args.byz_type == 'label_flip':
            byz = no_byz
        else:
            raise NotImplementedError

            # decide model architecture
        if args.net == 'cnn':
            net = cnn
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        elif args.net == 'fcnn':
            net = fcnn
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        elif args.net == 'mlr':
            net = MLR
            net.collect_params().initialize(mx.init.Xavier(magnitude=1.), force_reinit=True, ctx=ctx)
        else:
            raise NotImplementedError

        # define loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        # parameters
        num_workers = args.nworkers
        lr = args.lr
        epochs = args.nepochs
        grad_list = []
        old_grad_list = []
        weight_record = []
        grad_record = []
        train_acc_list = []

        # generate a string indicating the parameters
        paraString = str(args.dataset) + "+bias " + str(args.bias) + "+net " + str(
            args.net) + "+nepochs " + str(args.nepochs) + "+lr " + str(
            args.lr) + "+batch_size " + str(args.batch_size) + "+nworkers " + str(
            args.nworkers) + "+nbyz " + str(args.nbyz) + "+byz_type " + str(
            args.byz_type) + "+aggregation " + str(args.aggregation) + ".txt"

        # set up seed
        seed = args.seed
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # load dataset
        if args.dataset == 'mnist':
            if args.net == 'mlr':
                def transform(data, label):
                    return data.astype(np.float32) / 255, label.astype(np.float32)

                train_data = mx.gluon.data.DataLoader(
                    mx.gluon.data.vision.datasets.MNIST(train=True, transform=transform), 60000, shuffle=True,
                    last_batch='rollover')
                test_data = mx.gluon.data.DataLoader(
                    mx.gluon.data.vision.datasets.MNIST(train=False, transform=transform), 500, shuffle=False,
                    last_batch='rollover')

            elif args.net == 'cnn' or args.net == 'fcnn':
                def transform(data, label):
                    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

                train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                                      60000, shuffle=True, last_batch='rollover')
                test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 5000,
                                                     shuffle=False, last_batch='rollover')

        else:
            raise NotImplementedError

            # biased assignment
        bias_weight = args.bias
        other_group_size = (1 - bias_weight) / 9.
        worker_per_group = num_workers / 10

        # assign non-IID training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                if args.dataset == 'mnist' and args.net == 'cnn':
                    x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
                else:
                    x = x.as_in_context(ctx).reshape(-1, num_inputs)
                y = y.as_in_context(ctx)

                # assign a data point to a group
                upper_bound = (y.asnumpy()) * (1 - bias_weight) / 9. + bias_weight
                lower_bound = (y.asnumpy()) * (1 - bias_weight) / 9.
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()

                # assign a data point to a worker
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

        # concatenate the data for each worker
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

        # random shuffle the workers
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        # perform attacks
        if args.byz_type == 'label_flip':
            for i in range(args.nbyz):
                each_worker_label[i] = (each_worker_label[i] + 1) % 9
        if args.byz_type == 'backdoor':
            for i in range(args.nbyz):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1
                    each_worker_data[i][example_id][0][24][26] = 1
                    each_worker_data[i][example_id][0][26][24] = 1
                    each_worker_data[i][example_id][0][25][25] = 1
                    each_worker_label[i][example_id] = 0

        if args.byz_type == 'dba':
            for i in range(int(args.nbyz / 4)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz / 4), int(args.nbyz / 2)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][24][26] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz / 2), int(args.nbyz * 3 / 4)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][24] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz * 3 / 4), args.nbyz):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][25][25] = 1
                    each_worker_label[i][example_id] = 0

        ### begin training
        #set malicious scores
        malicious_score = []
        for e in range(epochs):
            # for each worker
            for i in range(100):
                # sample a batch
                with autograd.record():
                    output = net(each_worker_data[i][:])
                    loss = softmax_cross_entropy(output, each_worker_label[i][:])
                # backward
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])

            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]

            tmp = []
            for param in net.collect_params().values():
                tmp.append(param.data().copy())
            weight = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)

            # use lbfgs to calculate hessian vector product
            if e > 50:
                hvp = lbfgs(args, weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # perform attack
            if e%100 >= 50:
                param_list = byz(param_list, args.nbyz)

            if args.aggregation == 'trim':
                grad, distance = trim(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'simple_mean':
                grad, distance = simple_mean(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'median':
                grad, distance = median(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'krum':
                grad, distance = krum(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            else:
                raise NotImplementedError
            # Update malicious distance score

            if distance is not None and e > 50:
                malicious_score.append(distance)

            # update weight record and gradient record
            if e > 0:
                weight_record.append(weight - last_weight)
                grad_record.append(grad - last_grad)

            # free memory & reset the list
            if len(weight_record) > 10:
                del weight_record[0]
                del grad_record[0]

            last_weight = weight
            last_grad = grad
            old_grad_list = param_list
            del grad_list
            grad_list = []

            # compute training accuracy every 10 iterations
            if (e + 1) % 10 == 0:
                train_accuracy = evaluate_accuracy(test_data, net)
                if args.byz_type == 'backdoor' or 'dba':
                    backdoor_sr = evaluate_backdoor(test_data, net)
                    print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, train_accuracy, backdoor_sr))
                else:
                    print("Epoch %02d. Train_acc %0.4f" % (e, train_accuracy))

                train_acc_list.append(train_accuracy)

            # save the training accuracy every 100 iterations
            if (e + 1) % 100 == 0:
                if (args.dataset == 'mnist' and args.net == 'mlr'):
                    if not os.path.exists('mnist_mlr/'):
                        os.mkdir('mnist_mlr/')
                    np.savetxt('mnist_mlr/' + paraString, train_acc_list, fmt='%.4f')
                elif (args.dataset == 'mnist' and args.net == 'cnn'):
                    if not os.path.exists('mnist_cnn/'):
                        os.mkdir('mnist_cnn/')
                    np.savetxt('mnist_cnn/' + paraString, train_acc_list, fmt='%.4f')
                else:
                    raise NotImplementedError

            # compute the final testing accuracy
            if (e + 1) == args.nepochs:
                test_accuracy = evaluate_accuracy(test_data, net)
                print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
                with open('score1.csv', 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(malicious_score)

class Args:
    def __init__(self):
        self.dataset = 'mnist'            #dataset
        self.bias = 0.1                   #degree of non-IID
        self.net = 'cnn'                  #model network (Either 'mlr', 'cnn' or 'fcnn')
        self.batch_size = 32              #batch size
        self.lr = 0.0002                  #learning rate
        self.nworkers = 100               #number of clients
        self.nepochs = 100                #number of epochs
        self.gpu = -1                     #gpu 0 if using gpu, -1 using cpu
        self.seed = 41                    #seed
        self.nbyz = 28                    #number of poisoned clients
        self.byz_type = 'partial_trim'    #attack type ('no', 'partial_trim', 'full_trim', 'label_flip', 'backdoor', 'dba')
        self.aggregation = 'median'       #aggregation method ('simple_mean', 'trim', 'krum', 'median')

if __name__ == "__main__":
    args = Args()
    main(args)