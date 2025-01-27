# CS 444 Deepfake Detector using CNNs and LSTMs

### Instructions to Run Training Code

First, install processed frames that focus on the face from Google Drive (Note: ZIP file is ~30gb)

Make sure you have Python installed and install the gdown library:

```bash
pip install gdown
```

Then, install the .zip file from Google Drive using this command:

```bash
gdown 1GjX5bCPVRaXim2pmyGJM0ecWlejN65hO
```

Finally, unzip the .zip file:

```bash
unzip ./CELEB-DF-2.zip
```

Now, you can start the training process. To run the training code, you can use the shell script provided which sets up the variables to set as flags to call the python file. You can change the flags that vary the model parameters, such as opting in or out of Bi-LSTM and Attention Mechanism and changing the Sequence Length.


Run the bash script like this to start the python program:

```bash
bash ./demo.sh
```

### Instructions for Frame Extraction on a Different Dataset

If you have a different dataset and want to pre-process and extract the frames from the videos, you can execute this comand to do so (Note: Remember to replace 'NAME_OF_DIRECTORY_IN_HERE' with the path to the new data directory you want to pre-process):

```bash
python3 frame_extractor.py --data_dir="NAME_OF_DIRECTORY_IN_HERE"
```

### About evaluation.ipynb

The evaluation.ipynb file was used to run the final models and extract the results such as F1 Score, AUC-ROC Curve, Test Accuracy, and Confusion Matrix for evaluation in our report.


