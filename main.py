from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import glob
import numpy as np
import librosa
from scipy.io.wavfile import read
import joblib
import python_speech_features as mfcc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa.display
from PyQt5.QtCore import Qt, QItemSelectionModel


from PyQt5 import QtCore, QtGui, QtWidgets
import sounddevice as sd

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import task5UImanar
from scipy.special import softmax
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QLabel





class MainApp(QtWidgets.QMainWindow, task5UImanar.Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)


        self.features= []
         # Connect button to method
        self.pushButton_record.clicked.connect(self.toggle_record)

        # Set default image for the button
        default_image = QtGui.QPixmap(r"F:\task005\icons\record.png")
        self.pushButton_record.setIcon(QtGui.QIcon(default_image))

        # Create a figure and a canvas
        self.figure = Figure(figsize=(7,3))
        self.canvas = FigureCanvas(self.figure)

        # Add the canvas to the widget_spectrogram
        layout = QtWidgets.QVBoxLayout(self.widget_spectrogram)
        layout.addWidget(self.canvas)

        #self.actionVoice_fingerprint.triggered.connect(self.predict_word)

        self.label_5_ac_2.setText("Who is it?")


        self.current_mode = "Fingerprint"   # Variable to store the current mode
        self.isAccessDenied = False

       # Connect actions to methods
        actions_modes = {
            self.actionvoice_code: "Fingerprint",
            self.actionVoice_fingerprint: "Code"
        }

        for action, mode in actions_modes.items():
            action.triggered.connect(lambda checked, mode=mode: self.set_mode_and_predict(mode))

        
        
        # Connect itemClicked signal to handle_item_clicked method
        self.listWidget_persons.itemClicked.connect(self.handle_item_clicked)

        
        
        

        image_path_5_ac_2 = r'F:\task\icons\door (2).png'  # Replace with the actual image path
        self.set_image_in_label(self.label_status, image_path_5_ac_2)

        for i in range(min(4, self.listWidget_persons.count())):
            item = self.listWidget_persons.item(i)
            item.setCheckState(QtCore.Qt.Checked)
            self.listWidget_persons.setCurrentItem(item, QItemSelectionModel.Select)

    def set_mode_and_predict(self, mode):
            self.current_mode = mode
            if self.current_mode == "Code":
                self.predict_word()
            else:
                self.predict_speaker()

    def handle_item_clicked(self, item):
        # Toggle the check state of the item when it's clicked
        item.setCheckState(QtCore.Qt.Checked if item.checkState() == QtCore.Qt.Unchecked else QtCore.Qt.Unchecked)

    def set_image_in_label(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

    def get_image_path(self, winner_word, mode):
        image_folder = r'F:\task005\icons'  # Replace with the folder containing your images

        if not self.isAccessDenied and ((winner_word in [0, 1, 2]) or (winner_word == 3 and mode == "Fingerprint")):
            return f"{image_folder}\\door (3).png"
        else:
            return f"{image_folder}\\door (2).png"  # Replace with a default image path

        


    def toggle_record(self):
        # Change the button icon when clicked
        clicked_image = QtGui.QPixmap(r"F:\task005\icons\microphone.png")
        self.pushButton_record.setIcon(QtGui.QIcon(clicked_image))

        # Use QTimer to schedule recording after button icon is changed
        QtCore.QTimer.singleShot(5, self.record_audio)

    def record_audio(self):
        fs = 44100  # Sample rate
        seconds = 3  # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)  # Save as WAV file
        data, sr = librosa.load('output.wav', sr=fs)

        # # Convert stereo audio to mono
        # myrecording_mono = np.mean(myrecording, axis=1)

        # Restore the default image after recording is finished
        default_image = QtGui.QPixmap(r"F:\task005\icons\record.png")
        self.pushButton_record.setIcon(QtGui.QIcon(default_image))
        data = self.remove_noise(data)
        self.extract_features(data, sr)
        (self.predict_word if self.current_mode == 'Code' else self.predict_speaker)()

    def remove_noise(self,data):
        data= librosa.effects.preemphasis(data)
        return data
    def extract_features(self, data, sr):

        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
        delta = librosa.feature.delta(mfcc)
        delta_2 = librosa.feature.delta(delta)
        combined_feat = np.concatenate((mfcc, delta, delta_2))

        self.features.append(combined_feat.T)
        self.plot_spectrogram(data, sr)

        return combined_feat.T

    def plot_spectrogram(self, data, sr):
        # Clear the figure
        self.figure.clear()

        # Create an axes
        ax = self.figure.add_subplot(111)

        # Plot the spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        img=librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        plt.colorbar(img, format='%+2.0f dB')  # Specify the mappable object for the colorbar

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        self.figure.tight_layout()
        # Redraw the canvas
        self.canvas.draw()


    def predict_word(self):
        #self.current_mode = "Code"

        y=[]
        # Prediction code for word recognition
        audio, sr = librosa.load('output.wav')
        S = np.abs(librosa.stft(audio))

        model_folder2 = r'F:\task\2'

        # load the model of the words
        gmm_files_word = [f'{model_folder2}/{i}.joblib' for i in ['open', 'grant', 'unlock','others']]
        models_word = [joblib.load(fname) for fname in gmm_files_word]
        x_word = self.extract_features(audio, sr)


        # loop on the models of the words to get the max score of the word 
        log_likelihood_word = np.zeros(len(models_word)) 
        for j in range(len(models_word)):
            gmm = models_word[j] 
            scores = np.array(gmm.score(x_word))
            log_likelihood_word[j] = scores.sum()
        y.append(log_likelihood_word)
        probabilities = softmax(y)
        print(probabilities*100)


        winner_word = np.argmax(log_likelihood_word)
        print(log_likelihood_word)

 # function that get the id of the word 
        word_recognition_dict = {
            0: "Open middle door",
            1: "Grant me access",
            2: "Unlock the gate",
            3: "Access denied"
        }

        word_recognition = word_recognition_dict.get(winner_word, "Access denied")
        if word_recognition == "Access denied":
            self.isAccessDenied = True
        else:
            self.isAccessDenied = False
        if self.current_mode == "Fingerprint":
            return
        print(word_recognition)
        self.plot_pie_chart(probabilities, ["Open ", "Grant ", "Unlock ", "Access Denied"])


        # Display the final statement in the label
        final_statement = word_recognition
        self.label_5_ac_2.setText(final_statement)
        # Update the image in label_5_ac_2
        new_image_path = self.get_image_path(winner_word, self.current_mode)
        self.set_image_in_label(self.label_status, new_image_path)




    def predict_speaker(self):
            #self.current_mode = "Fingerprint"
           
        y_word = []
        y_speaker = []
        
        audio, sr = librosa.load('output.wav')

        # Prediction code for word recognition
        model_folder2 = r'F:\task\2'
        gmm_files_word = [f'{model_folder2}/{i}.joblib' for i in ['open', 'grant', 'unlock', 'others']]
        models_word = [joblib.load(fname) for fname in gmm_files_word]
        x_word = self.extract_features(audio, sr)
        
        log_likelihood_word = np.zeros(len(models_word))
        for j in range(len(models_word)):
            gmm = models_word[j]
            scores = np.array(gmm.score(x_word))
            log_likelihood_word[j] = scores.sum()
        y_word.append(log_likelihood_word)
        probabilities_word = softmax(y_word)
        

        # Prediction code for speaker recognition
        model_folder = r'F:\task\1'
        gmm_files = [f'{model_folder}/{i}.joblib1' for i in ['manar', 'salma', 'sara', 'yasmeen', 'others']]
        models = [joblib.load(fname) for fname in gmm_files]
        x_speaker = self.extract_features(audio, sr)

        log_likelihood_speaker = np.zeros(len(models))
        for j in range(len(models)):
            gmm = models[j]
            scores = np.array(gmm.score(x_speaker))
            log_likelihood_speaker[j] = scores.sum()
        y_speaker.append(log_likelihood_speaker)
        probabilities_speaker = softmax(y_speaker)
        winner = np.argmax(log_likelihood_speaker)

        # check the id of the persons 
        speakers = ['Manar', 'Salma', 'Sarah', 'Yasmeen', 'Others']
        speaker = speakers[winner]
        # Check if the speaker is in the listWidget_persons and is selected
        for i in range(self.listWidget_persons.count()):
            item = self.listWidget_persons.item(i)
            if item.text() == speaker and item.checkState() == QtCore.Qt.Checked:
                break
        else:
            speaker = "Others"
            winner = 4
        print(speaker)

        word_winner = np.argmax(log_likelihood_word)
        word_labels = ["Open", "Grant", "Unlock", "Denied"]
        word_recognition = word_labels[word_winner]

        self.plot_pie_charts(probabilities_word, word_labels,
                             probabilities_speaker, speakers)

        self.predict_word()


        # Update the image in label_5_ac_2
        new_image_path = self.get_image_path(winner, self.current_mode)
        self.set_image_in_label(self.label_status, new_image_path)
        # Display the final statement in the label
        final_statement = f"Speaker: {speaker}, Word: {word_recognition}"
        self.label_5_ac_2.setText(final_statement)

        
    
    def plot_pie_chart(self, probabilities, labels):
        # Ensure probabilities sum to 1
        probabilities = probabilities / np.sum(probabilities)

        fig = Figure(figsize=(3, 3), dpi=100)
        ax = fig.add_subplot(111)

        # Flatten the probabilities array
        probabilities = probabilities.flatten()

        ax.pie(probabilities, labels=labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'])

        canvas = FigureCanvas(fig)
        canvas.draw()

        scene = QtWidgets.QGraphicsScene(self)
        scene.addWidget(canvas)
        self.graphicsView_similarity.setScene(scene)




    def plot_pie_charts(self, probabilities_word, labels_word, probabilities_speaker, labels_speaker):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Plot pie chart for word recognition
        axes[0].pie(probabilities_word.flatten(), labels=labels_word, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        axes[0].set_title("Word Recognition")

        # Plot pie chart for speaker recognition
        axes[1].pie(probabilities_speaker.flatten(), labels=labels_speaker, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'])
        axes[1].set_title("Speaker Recognition")

       

        canvas = FigureCanvas(fig)
        canvas.draw()

        scene = QtWidgets.QGraphicsScene(self)
        scene.addWidget(canvas)
        self.graphicsView_similarity.setScene(scene)







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())