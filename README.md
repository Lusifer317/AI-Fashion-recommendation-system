
# Fashion Recommender System

This project is a **Fashion Recommender System** that utilizes a transfer learning model with **ResNet50** for feature extraction and **K-Nearest Neighbors (KNN)** for recommending visually similar items. The system is deployed using **Streamlit** for an interactive and user-friendly experience. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

### Project Overview
This Fashion Recommender System was developed to provide users with recommendations based on visual similarity between items. The system was trained on a dataset of 44,000 fashion images, leveraging **ResNet50** for transfer learning and **KNN** for finding similar images.

### Features
- **Image-Based Recommendations**: Recommends items visually similar to the selected image.
- **Transfer Learning**: Uses ResNet50 pre-trained on ImageNet for efficient feature extraction.
- **Interactive Interface**: Deployed as a web application with Streamlit for easy user interaction.

### Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fashion-recommender-system
   ```

2. **Install required packages:**
   Ensure you have Python 3.8+ installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   Place your dataset of fashion images in a folder named `data/images/` in the project directory. Update the paths if necessary in the code.

4. **Run the Application:**
   Launch the Streamlit application with the following command:
   ```bash
   streamlit run app.py
   ```

### Usage
1. **Start the Application**: Run the Streamlit application (instructions above).
2. **Upload or Select an Image**: Use the interface to upload or select a fashion image.
3. **Get Recommendations**: The system will display visually similar items based on the uploaded image.

### Project Structure
```
fashion-recommender-system/
├── data/
│   └── images/                  # Folder for fashion images
├── model/
│   ├── resnet50_feature_extractor.py   # Script to extract features using ResNet50
│   └── knn_recommender.py       # KNN recommender logic
├── app.py                       # Streamlit app for deployment
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```

### Model Architecture
The recommendation system utilizes a **ResNet50** model pre-trained on ImageNet to extract visual features. These features are used to train a **K-Nearest Neighbors (KNN)** model, which identifies visually similar items.

#### Steps in Model Pipeline:
1. **Feature Extraction**: ResNet50 is used to extract 2048-dimensional feature vectors for each image in the dataset.
2. **Similarity Search**: A KNN model (with Euclidean distance metric) is applied to find items similar to the query image.

### Results
The model provides accurate recommendations for items visually similar to the selected image. Performance was evaluated based on recommendation relevance and user satisfaction.

### Future Improvements
- **Enhanced Recommendation Algorithms**: Experiment with more advanced techniques such as Siamese networks for similarity learning.
- **User Feedback Integration**: Implement a feedback system to refine recommendations based on user preferences.
- **Multimodal Recommendations**: Incorporate text data (e.g., product descriptions) along with image features for richer recommendations.
