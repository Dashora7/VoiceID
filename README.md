# VoiceID
A Repository for Voice ID and Authentication

Add your voice files to the data section under a file with your name.

Use processing.py to record your voice and save a sample

Use make_voice_clf.ipynb to parse through your voice and generate embeddings using a neural voice encoder. You can additionally generate embeddings for random or "unauthorized" voices as mined from the OpenVoice dataset. Your may analyze these mebeddings by showing the respective distances from your voice centroid. You can then train a Perceptron to predict you voice using the embeddings that are yours and the ones that aren't

Run clf_utils to encapsulate this whole process. Just create a VoiceIDSystem object, add yourself as an authorized user along with your voice data, and the system will add your voice emdeddings. Then, you can run a simple "id_subject" command on the system to record yourself and identify the identity of the speaker based on the added embeddings.
