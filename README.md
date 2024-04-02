# Face Mask Detection Project

This project is designed to detect whether a person is wearing a face mask or not using machine learning techniques. It utilizes the TensorFlow and OpenCV libraries to build and train a convolutional neural network (CNN) model for classification.

## Project Structure

The project is structured as follows:

- `README.md`: This file contains information about the project and how to set it up.
- `main.py`: Python script containing the main code for training the CNN model and performing face mask detection.
- `requirements.txt`: File listing all the required Python packages for the project.
- `weights.caffemodel`: Pre-trained Caffe model used for face detection in images.
- `Face Mask Dataset/`: Directory containing the face mask dataset split into training, testing, and validation sets.
- `utils.py`: Utility functions used in the project, such as image preprocessing and data loading.
- `model/`: Directory containing saved models after training.
- `images/`: Directory containing sample images for testing the face mask detection model.

## Getting Started

1. Clone the repository:
2. 2. Install the required packages:
3. Download the Face Mask Dataset and place it in the appropriate directory (`Face Mask Dataset/`).
4. Run the `main.py` script to train the model and perform face mask detection.

## Usage

- To train the model, run:
- To perform face mask detection on sample images, run:

## Acknowledgements

- The Face Mask Dataset used in this project is sourced from [source-name].
- The pre-trained Caffe model (`weights.caffemodel`) for face detection is obtained from [source-name].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
