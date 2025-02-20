import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm

# This will Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# This will Load MobileNetV2 Model
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# This will Extract Features Function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# This will Ensure images folder exists
image_folder = "images"
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"❌ Folder '{image_folder}' not found! Place images inside it.")

# This will Get Image Filenames
filenames = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

# This will Extract and Save Features
feature_list = np.zeros((len(filenames), 1280), dtype=np.float16)

for i, file in tqdm(enumerate(filenames), total=len(filenames), desc="Extracting Features"):
    feature_list[i] = extract_features(file, model)

# This will Save Features and Filenames
np.save("embeddingss.npy", feature_list)
pickle.dump(filenames, open("filenamess.pkl", "wb"))

print(" Feature extraction completed successfully!")


