# Personal-Projects

## EVA - Emotional Voice Analyzer

EVA (Emotional Voice Analyzer) is a machine learning-powered web application designed to analyze the emotional tone in voice recordings. This project aims to help users understand the emotions conveyed through tone of voice.

### Table of Contents
- [Description](#description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Acknowledgements](#acknowledgements)

### Description
EVA utilizes a machine learning model created with TensorFlow and Keras to predict emotions from audio files. The model is trained on a dataset of voice recordings with varying sizes and emotional tones. The web application, built with Flask, allows users to upload audio files, which are then processed to extract features and predict the emotional tone.

#### Limitations
- The application currently supports audio files up to 10 seconds in length.
- Supported audio formats are .mp3, .wav, and .m4a.

### Features
- **Machine Learning Algorithm**: Built with TensorFlow and Keras.
- **Real-time Predictions**: Upload audio files and get instant predictions.
- **User-friendly Interface**: Simple and intuitive web interface for uploading and analyzing audio.

### Technologies Used
- Python
- Flask
- TensorFlow
- Keras
- MySQL
- HTML
- CSS
- JavaScript

### Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/FrankieVara/Personal-Projects.git
    cd Personal-Projects/CSCI490Final/
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Set up the MySQL server using XAMPP:
    - Install and configure XAMPP to run the MySQL server.
    - Update the database connection settings in `app.py` to match your local configuration.

4. Run the application:
    ```bash
    python app.py
    ```

### Usage
1. Upload an audio file through the web interface.
2. The file path is saved, and the audio features are extracted.
3. The model processes the audio to predict the emotional tone.

### Screenshots
Here are some screenshots of the application in action:

#### Home Page
![Home Page](./screenshots/Screenshot1.png)

#### File Upload
![File Upload](./screenshots/Screenshot2.png)

#### Prediction
![Prediction](./screenshots/Screenshot3.png)

## Acknowledgements
- Special thanks to the contributors of the audio dataset on Kaggle.
- Thanks to the developers of TensorFlow, Keras, and Flask for their amazing libraries.

