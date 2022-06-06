from difflib import SequenceMatcher
from errno import EPERM
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
import matplotlib.pyplot as plt
from argparse import ArgumentParser


# Hyperparameters (Default setting)
# EPOCH = 5
# BATCH_SIZE = 128
# LR = 0.01
# SAVE_MODEL = True
# LOAD_MODEL = False
# SAVE_WEIGHT_PATH = 'weights/epoch-'
# LOAD_WEIGHT_PATH = 'weights/epoch/checkpoint.ckpt'

# ArgumentParser
parser = ArgumentParser()
parser.add_argument("--EPOCH", default=5, type=int)
parser.add_argument("--BATCH_SIZE", default=128, type=int)
parser.add_argument("--LR", default=0.01, type=float)
parser.add_argument("--SAVE_MODEL", default=True, type=bool)
parser.add_argument("--LOAD_MODEL", default=False, type=bool)
parser.add_argument("--SAVE_WEIGHT_PATH", default='weights/epoch-', type=str)
parser.add_argument("--LOAD_WEIGHT_PATH", default='weights/epoch-0', type=str)
args = parser.parse_args()

EPOCH = args.EPOCH
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR
SAVE_MODEL = args.SAVE_MODEL
LOAD_MODEL = args.LOAD_MODEL
SAVE_WEIGHT_PATH = args.SAVE_WEIGHT_PATH
LOAD_WEIGHT_PATH = args.LOAD_WEIGHT_PATH


# Check if the system get the GPU.
if tf.config.list_physical_devices('GPU'):
    print('\nUsing GPU for training (つ´ω`)つ\n')
else:
    print('\nNo GPU! 。･ﾟ･(つд`)･ﾟ･\n')


# Load the data.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Visualize some samples.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i+10], cmap='binary')
    plt.xlabel(class_names[train_labels[i+10]])
plt.show()


x_train = train_images.astype("float32") / 255.0
x_test = test_images.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  # (60000, 28, 28) → (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]    # (10000, 28, 28) → (10000, 28, 28, 1)

# from_tensor_slices → create an "Dataset Object"
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_labels))
test_dataset = test_dataset.batch(BATCH_SIZE)


# Custom Model
class SimpleModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv2D(32, kernel_size = (3,3), input_shape = (None, 28, 28, 1),  activation="relu", name = 'input_layer')
        self.maxpool_1 = MaxPool2D((2,2))
        self.bn_1 = BatchNormalization()
        self.ftn_1 = Flatten()
        self.fc_1 = Dense(10, activation = 'softmax', name = 'output_layer')

    def call(self, input):
        x = self.conv_1(input)
        x = self.maxpool_1(x)
        x = self.bn_1(x)
        x = self.ftn_1(x)
        output = self.fc_1(x)

        return output
    
    def model(self):
        x = Input(shape=(28,28,1))
        return tf.keras.Model(x, self.call(x))


model = SimpleModel()


if LOAD_MODEL:
    print('loading weights...')
    model.load_weights(LOAD_WEIGHT_PATH + '/checkpoint.ckpt')


model.model().summary()


# sgd = SGD(learning_rate=0.001)   
# rms = RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)
# adg = Adagrad(learning_rate=0.001, epsilon=None, decay=0.0)
optimizer = Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


loss_fn = SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

train_loss_record = []
train_accuracy_record = []
test_loss_record = []
test_accuracy_record = []

print("\nTraining starts :)")

for epoch in range(EPOCH):
    print('=======================================================================================================================================================================================')
    tqdm_bar = tqdm(train_dataset)
    
    for idx, (images, labels) in enumerate(tqdm_bar):
        with tf.GradientTape() as tape:
            # forward pass to get predictions
            predictions = model(images)
            # compute loss with the ground and our predictions
            loss = loss_fn(labels, predictions)
        # compute gradient 
        gradients = tape.gradient(loss, model.trainable_variables)
        # backpropagation
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        
        train_accuracy_display = round(float(train_accuracy.result()), 4)
        train_loss_display = round(float(train_loss.result()), 4)

        tqdm_bar.set_description(f'Epoch [{epoch + 1}/{EPOCH}]')
        if epoch > 0:
            tqdm_bar.set_postfix(
                                 test_accuracy = test_accuracy_display,
                                 test_loss = test_loss_display,
                                 train_accuracy = train_accuracy_display,
                                 train_loss = train_loss_display,
                                )     
        else:
            tqdm_bar.set_postfix(
                                 train_accuracy = train_accuracy_display, 
                                 train_loss = train_loss_display,
                                )

    train_loss_record.append(train_loss_display)
    train_accuracy_record.append(train_accuracy_display)

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for test_imgs, test_labels in test_dataset:
        # forward pass to get predictions
        predictions = model(test_imgs)
        # compute loss with the ground and our predictions
        t_loss = loss_fn(test_labels, predictions)

        test_loss(t_loss)
        test_accuracy(test_labels, predictions)

    test_loss_display = round(float(test_loss.result()), 4)
    test_accuracy_display = round(float(test_accuracy.result()), 4)

    if SAVE_MODEL:
        print('saving weights...')
        model.save_weights(SAVE_WEIGHT_PATH + str(epoch + 1) + '/checkpoint.ckpt')
        

print("\ntest sample...")

test_sample = x_test[1:2]
output = int(tf.argmax(model(test_sample), 1))
print('The sample class is: ', class_names[output])
plt.axis(False)
plt.imshow(test_images[1], cmap='binary')
plt.show()

plt.figure(0)
plt.subplot(121)
plt.plot(range(len(train_loss_record)), train_loss_record,label='loss')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(train_accuracy_record)), train_accuracy_record,label='accuracy')
plt.title('Accuracy')
plt.savefig('custom_training.png',dpi=300,format='png')
plt.show()
print('Result saved into custom_training.png')