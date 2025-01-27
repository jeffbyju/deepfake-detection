# CS 444 Deepfake Detector using CNNs and LSTMs

First, unzip the models in the models directory

```bash
unzip models/models.zip
```

Then, to run the flask application, simply run this command in your terminal

```bash
python3 app.py --num_frames=20
```

Use the videos in 'Test Data' directory from the root directory to upload into the Flask application to run the demo. 

The CELEB-DF dataset has 3 sub-datasets, Youtube real, Celeb real, and Celeb fake, each of them have videos.

### Video Key:
File Name | Subset Name | Class Name | Model Result Class
--- | --- | --- | --- |
00097.mp4 | Youtube Real Subset | Real Class | Real Class
00121.mp4 | Youtube Real Subset | Real Class | Fake Class
id29_0000.mp4 | Celeb Real Subset | Real Class | Fake Class
id35_0007.mp4 | Celeb Real Subset | Real Class | Real Class
id34_id38_0005.mp4 | Celeb Fake Subset | Fake Class | Fake Class
id61_id60_0009.mp4 | Celeb Fake Subset | Fake Class | Fake Class