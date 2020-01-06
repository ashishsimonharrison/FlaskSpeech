import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
from IPython.display import Audio
import wave
import sample_models as sm
model_end = sm.final_model(input_dim=13,
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=250,
                        activation='tanh',
                        cell=sm.GRU,
                        dropout_rate=1,
                        number_of_layers=3)


def get_predictions_rec(input_to_softmax, a_path, model_path):
    data_gen = AudioGenerator(spectrogram=False)
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    audio_path = a_path
    data_point = data_gen.normalize(data_gen.featurize(audio_path))

        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    
    
    return 'Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints))
    


from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('index.html')
	
@app.route('/uploader')
def upload_file1():
      #print(request.files['audio_data'].filename)
      
      
      abc = "output.wav"
      text = get_predictions_rec(input_to_softmax=model_end, a_path=abc,
               model_path='results/model_end.h5')
      return render_template('abc.html',text = text)

@app.route('/messages', methods = ['POST','GET'])
def api_message():
   
   f = open('./file.wav', 'wb')
   f.write(request.data)
   f.close()
   return "Done"

if __name__ == '__main__':
   app.run(debug = False, threaded=False)