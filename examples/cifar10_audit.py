import tensorflow as tf
import sys
import os

# Add parent directory to path so the 'tensorspeed' package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorspeed.auditor import Audit

# Define a simple CNN for CIFAR-10 instead of loading a missing .h5 file
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # optional: print model architecture

# Load and preprocess CIFAR-10
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0

# Train the model → creates the history object needed for plotting
print("Training model for 5 epochs...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

# Initialize Audit and plot learning curves
print("Running audit and plotting learning curves...")
auditor = Audit(model=model)           # change to Audit(model=model, history=history) if needed
auditor.plot_learning_curves(history)
