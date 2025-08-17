import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import models
import os

# Load the pre-trained MobileNetV2 model with weights from ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a few layers on top for classification (fine-tuning)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)  # 2 classes: TB and Non-TB
model = models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model for fine-tuning
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load your trained model weights if available (for TB classification)
# model.load_weights('path_to_your_model_weights.h5')

# Grad-CAM function
def grad_cam(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap[0]

# Function to process and predict an image
def predict_and_visualize(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # Resize to model input size
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_names = ['Non-TB', 'TB']  # Update this according to your dataset

    print(f"Predicted class: {class_names[predicted_class[0]]}")

    # Get the Grad-CAM heatmap
    heatmap = grad_cam(model, img_array)

    # Resize the heatmap to the original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to a colormap (apply color to heatmap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    # Plot the original image, heatmap, and superimposed image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title("Grad-CAM Heatmap")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Superimposed Image")

    plt.show()

# Example usage
image_path = '/content/drive/MyDrive/Tbdet/tb.jpeg'  # Path to your chest X-ray image
predict_and_visualize(image_path)
