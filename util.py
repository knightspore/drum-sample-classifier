import os
import pandas as pd
import librosa
import librosa.display
from matplotlib import pyplot as plt
from audio_util import AudioUtil
import torch
from torch.utils.data import Dataset
import time

# Useful Class References
classes = {
    "kick": 0,
    "snare": 1,
    "hat": 2,
    "clap": 3,
    "ride": 4,
    "crash": 5,
    "tom": 6,
    "perc": 7,
    "cowbell": 8,
    "clave": 9,
}

classes_reverse = {
    0: "kick",
    1: "snare",
    2: "hat",
    3: "clap",
    4: "ride",
    5: "crash",
    6: "tom",
    7: "perc",
    8: "cowbell",
    9: "clave",
}



# WavDataSet Takes a Path and (badly) creates a dataset from all .wav samples found there.
class WavDataSet:
    def __init__(self, path):
        self.classes = {
        "kick": [],
        "snare": [],
        "hat": [],
        "clap": [],
        "ride": [],
        "crash": [],
        "tom": [],
        "perc": [],
        "cowbell": [],
        "clave": [],
        }
        self.classID = {
            "kick": 0,
            "snare": 1,
            "hat": 2,
            "clap": 3,
            "ride": 4,
            "crash": 5,
            "tom": 6,
            "perc": 7,
            "cowbell": 8,
            "clave": 9,
        }
        # Populate Classes
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".wav") and "LOOP" not in file.upper():
                    for k in list(self.classes.keys()):
                        if k.upper() in file.upper():
                            self.classes[k].append({
                        "path": os.path.join(root, file),
                        "filename": file,
                    })

    def __repr__(self): # Print Summary
        list = ""
        for _class in self.classes:
            list += f"{_class}: {len(self.classes[_class])}\n"
        return list

    def csv(self, path): # Save CSV
        df = pd.DataFrame()

        for _class in self.classes:
            for sample in self.classes[_class]:
                df = df.append({
                    "sample_file_name": sample["filename"],
                    "path": sample["path"].replace("\\", "/"),
                    "classID": _class,
                    "class": self.classID[_class],
                }, ignore_index=True)

        df.to_csv(path, index=False)

# Interfact Between Dataset and DataLoader
class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        self.duration = 4000
        self.sr = 44100
        self.channel = 2

    # No. Items in the Dataset
    def __len__(self):
        return len(self.df)
    
    # Get i'th item
    def __getitem__(self, idx):
        audio_file = self.df.iloc[idx]['path']
        class_id = self.df.iloc[idx]['class']

        # This is where we re-process samples to equalize them
        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel) 
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration) # sig, sr
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
				# plot_spectrogram(sgram[0], title=f"Random Sample Spec: {os.path.basename(audio_file)}")
				# plot_spectrogram(aug_sgram[0], title=f"Random Augmented Spec: {os.path.basename(audio_file)}")
        
        return aug_sgram, class_id, idx

# Single File Prediction Function

def predict(model, path, classID, device):
    model.eval()
    
    # Prepare Prompt as Faked Dataset
    prompt = { "path": [path], "class": classes[classID] }
    p_df = pd.DataFrame(data=prompt)
    pred_ds = SoundDS(p_df)
    pred_dl = torch.utils.data.DataLoader(pred_ds, batch_size=1, shuffle=False)
    
    # Disable Grad
    with torch.no_grad():
        for data in pred_dl:
            start_time=time.time()
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize Inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            # Get Predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            correct = (prediction == labels).sum().item()
            
            filename = os.path.basename(path)
            yn = "✅" if correct == 1 else "❌"
            guess = classes_reverse[prediction.cpu().numpy()[0]]
            
            print(f"{yn} {filename} {'is a' if correct == 1 else 'is NOT a'} {guess} ({time.time() - start_time:.2f}s)")

# Plot a Mel Spec in PyPlot

def PlotSpectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

