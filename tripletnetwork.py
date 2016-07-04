import lasagne
import numpy as np
import theano
import theano.tensor as T
import time


# Loss function.
def loss_fn(network_output, size):
    """
    Compute the triplet network loss.
    :param network_output: T.tensor
    :type size: int
    :return:
    """
    exp_p = T.exp(((network_output[:size] - network_output[size:2*size]) ** 2).sum(axis=1))
    exp_n = T.exp(((network_output[:size] - network_output[2*size:]) ** 2).sum(axis=1))
    denom = exp_p + exp_n
    dist_p = exp_p / denom
    return (dist_p ** 2).mean()


def generate_dataset(n):
    raise NotImplementedError


def generate_minibatches(dataset, batchsize=128):
    d = dataset[0][0].size
    for i in range(0, len(dataset), batchsize):
        X = np.empty(3*batchsize, d)
        for x in dataset[i:i+batchsize]:
            X[i, :] = x[0]
            X[2*batchsize+i, :] = x[1]
            X[3*batchsize+i, :] = x[2]
    yield X


def build_mlp(input_var=None, input_size=4096):
    l_in = lasagne.layers.InputLayer(shape=(None, input_size), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    return l_out


def learn(dataset, model='mlp', num_epochs=10, batchsize=128):
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    else:
        print("Unrecognized model type %r." % model)
    prediction = lasagne.layers.get_output(network)
    train_loss = loss_fn(prediction, batchsize)

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         train_loss, params, learning_rate=0.5, momentum=0.9)
    updates = lasagne.updates.adam(train_loss, params)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = loss_fn(test_prediction, batchsize)


    # # As a bonus, also create an expression for the classification accuracy:
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                   dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var], train_loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in generate_minibatches(dataset, batchsize, 'train'):
            if batch is not None:
                train_err += train_fn(batch.todense())
                train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in generate_minibatches(dataset, batchsize, 'val'):
            if batch is not None:
                val_err += val_fn(batch.todense())
                val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #     val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', weights=lasagne.layers.get_all_param_values(network))
    # np.savez('projections.npz', test=project(X_test, input_var, network))


def main():
    dataset = generate_dataset(n=10000)
    learn(dataset, batchsize=256, num_epochs=32)


if __name__ == '__main__':
    main()


