

def load_image_list(train=True):
    train_path = ''
    test_path = ''
    if train:
        data_addresses = glob.glob(train_path + '*.png')
    else:
        data_addresses = glob.glob(test_path + '*.png')
    return data_addresses

def randomize_data(data_addresses):
    return np.random.permutation(data_addresses)

def load_addr_batch(data_addresses, batch_size):
    batch_addr = np.random.choice(data_addresses, size=batch_size)
    return batch_addr

def load_image_batch(data_addresses):
    '''print([np.array(Image.open(fname)).shape for fname in data_addresses])'''
    batch = np.array([np.array(Image.open(fname)) for fname in data_addresses], dtype=np.float32)
    # normalizing images
    #batch /= 255.0
    return batch

def create_labels(data_addresses, bin_encoding=False):
    digits = '0123456789_'
    classes = [address.split('/')[-1].split('.')[0].strip(digits) for address in data_addresses]
    #print(labels) for debugging
    if bin_encoding:
        labels = [0 if 'car' in cl else 1 for cl in classes]
    return np.array(labels, dtype=np.float32)[:, np.newaxis], classes

def one_hot_encode(data_classes):
    m = len(data_classes)
    one_hot_encoded_data = np.zeros((m, 2))
    for i in range(m):
        if data_classes[i] == 'car':
            one_hot_encoded_data[i] = [1, 0]
        else:
            one_hot_encoded_data[i] = [0, 1]
    return np.array(one_hot_encoded_data)

def load_random_train_batch(batch_size=32):
    # load image addresses
    data_addresses = load_image_list()
    # randomize them
    rand_addresess = randomize_data(data_addresses)
    # load 'batch_size' images
    batch_addresses = load_addr_batch(rand_addresess, batch_size)
    # load images
    X_data = load_image_batch(batch_addresses)
    # create labels
    Y_labels, classes  = create_labels(batch_addresses, bin_encoding=True)
    Y_ohe = one_hot_encode(classes)
    return X_data, Y_labels, Y_ohe

def load_random_test_batch(n_images=1):
    data_addresses = load_image_list(train=False)
    batch_addresses = load_addr_batch(data_addresses, n_images)
    # load images
    X_data = load_image_batch(batch_addresses)
    # create labels
    Y_labels, classes = create_labels(batch_addresses, bin_encoding=True)
    Y_ohe = one_hot_encode(classes)
    return X_data, Y_labels, Y_ohe

def load_train_batch(start, end):
    # load image addresses
    data_addresses = randomize_data(load_image_list())
    # load 'batch_size' images
    batch_addresses = data_addresses[start:end]
    # load images
    X_data = load_image_batch(batch_addresses)
    # create labels
    Y_labels, classes  = create_labels(batch_addresses, bin_encoding=True)
    Y_ohe = one_hot_encode(classes)
    return X_data, Y_labels, Y_ohe