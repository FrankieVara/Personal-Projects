from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import librosa

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/audio_paths?unix_socket=/Applications/XAMPP/xamppfiles/var/mysql/mysql.sock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



db = SQLAlchemy(app)

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    file_path = db.Column(db.String(512), unique=True, nullable=False)

    def __init__(self, filename, file_path):
        self.filename = filename
        self.file_path = file_path



from keras.models import Sequential, model_from_json
import os
import pickle
import librosa

json_file = open('/Users/franciscovara/Documents/CSCI490Final/Backend/kaggle/CNN_model_final.json', 'r')
loded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loded_model_json)
#load weights into new model
loaded_model.load_weights("/Users/franciscovara/Documents/CSCI490Final/Backend/kaggle/best_model1_weights_final.h5")
print ("loaded model")


with open('/Users/franciscovara/Documents/CSCI490Final/Backend/kaggle/scaler2_final.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('/Users/franciscovara/Documents/CSCI490Final/Backend/kaggle/encoder2_final.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data, frame_length=2048, hop_length=512):
  # Correctly passing `y=data` as a keyword argument
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    # Ensure to pass `y=data` as a keyword argument
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=hop_length)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)





def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    #result=np.array(res)
    shape = res.shape
    # If the number of features is less than 2376, pad with zeros
    if shape[0] < 2376:
        pad_width = ((0, 2376 - shape[0]))  # Pad along the second axis
        res = np.pad(res, pad_width, mode='constant', constant_values=0)

    result = np.reshape(res, newshape=(1, -1))  # Reshape to (1, 2376) or whatever the shape is after padding

    #scaler2 = StandardScaler(n_features=2376)
    #result=np.reshape(result,newshape=(1,shape[0]))
    #scaler2.fit(result)
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)

    return final_result


def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    files = request.files.getlist('file')
    for file in files:
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            audio_file = AudioFile(filename=filename, file_path=file_path)
            db.session.add(audio_file)
    
    db.session.commit()
    return render_template('file_uploaded.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract the path of the audio file from form data
        audio_file_path = request.form['audio_file']
        prediction_result = make_prediction(audio_file_path)
        return render_template('prediction_result.html', prediction=prediction_result)
    else:
        # Query your database for all available audio files
        audio_files = AudioFile.query.all()
        return render_template('predict.html', audio_files=audio_files)

def make_prediction(audio_file_path):
    # Presumably get_predict_feat() prepares the audio file for prediction
    res = get_predict_feat(audio_file_path)
    predictions = loaded_model.predict(res)
    # Use label encoder to translate prediction to human-readable form
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

@app.route('/check_database', methods=['GET'])
def get_articles():
    audio_files = AudioFile.query.all()
    data = [{"id": file.id, "filename": file.filename, "file_path": file.file_path} for file in audio_files]
    
    return jsonify({"audio_files": data})


if __name__ == "__main__":
    app.run(debug=True)

