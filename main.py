dir1_path = '/media/pc/40916c07-dc4a-4fff-ba73-2833d2abdc01/Black/scene_1_images/*.jpg'
dir2_path = '/media/pc/40916c07-dc4a-4fff-ba73-2833d2abdc01/Black/scene_2_images/*.jpg'
dir3_path = '/media/pc/40916c07-dc4a-4fff-ba73-2833d2abdc01/Black/scene_3_images/*.jpg'
test_img = '/media/pc/40916c07-dc4a-4fff-ba73-2833d2abdc01/Black/scene_3_images/image-00061.jpg'
dump_file = 'hash.dat'


# Feature extractor
def extract_features(image_path, vector_size=32):
    import cv2
    import numpy as np
    from scipy.misc import imread

    image = imread(image_path, mode="RGB")
    try:
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

        kps, desc = alg.compute(image, kps)
        desc = desc.flatten()

        needed_size = (vector_size * 64)
        if desc.size < needed_size:
            desc = np.concatenate([desc, np.zeros(needed_size - desc.size)])

    except cv2.error as e:
        print 'Error: ', e
        return None

    return kps, desc


def create_autoencoder(image_path):
    from keras.layers import Input, Dense
    from keras.models import Model

    # this is the size of our encoded representations
    encoding_dim = 32  # 64 floats -> compression of factor xxx, assuming the input is 784 floats

    input_img = Input(shape=(2048,))
    encoded = Dense(256, activation='relu')(input_img)
    encoded = Dense(128, activation='relu')(encoded)

    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(2048, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)                                                                             # autoencoder model, decodes and encodes
    encoder = Model(input_img, encoded)                                                                                 # encoder model, encodes to fixed size

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-3]
    decoder = Model(encoded_input, decoder_layer(encoded_input))                                                        # decoder model, decodes from fixed size to original

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(
                image_path, image_path,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(image_path, image_path)
    )

    return encoder, decoder


def classifier(train, labels, test):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
    knn.fit(train, labels)

    p = knn.predict(test)

    return p


def run():
    import glob
    import numpy as np

    labels = []
    paths_1 = glob.glob(dir1_path)
    for _ in paths_1: labels.append(True)
    paths_2 = glob.glob(dir2_path)
    for _ in paths_2: labels.append(False)
    paths_3 = glob.glob(dir3_path)
    for _ in paths_3: labels.append(False)

    paths = paths_1 + paths_2 + paths_3
    train_imgs = np.zeros((len(paths), 2048))

    for index, path in enumerate(paths):
        kps, desc = extract_features(path)
        train_imgs = np.insert(train_imgs, index, desc, 0)

    encoder, decoder = create_autoencoder(train_imgs)
    train = encoder.predict(train_imgs)

    train.dump(dump_file)

    _, desc = extract_features(test_img)
    desc = np.reshape(desc, (1, 2048))

    print np.shape(train)
    print np.shape(labels)
    print np.shape(desc)

    result = classifier(train, labels, desc)

    print result


run()