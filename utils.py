import re
import os
import librosa
import numpy as np


def draw_mel_gram(audio_file, ax):
    samples, sample_rate = librosa.load(audio_file, sr=None)
    sgram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax)

    # Name
    name = ' '.join(extract_anno(os.path.basename(audio_file)))
    ax.set(title=name)
    ax.label_outer()

def extract_anno(file_name):
    """
    We are using the samples from https://www.upf.edu/web/mtg/irmas
    Following the annotation from the samples, we have:
    [cel] : Cello
    [cla] : Clarinet
    [flu] : Flute
    [gac] : Acoustic guitar
    [gel] : Electric guitar
    [org] : Organ
    [pia] : Piano
    [sax] : Saxophone
    [tru] : Trumpet
    [vio] : Violin
    [voi] : Human Singing Voice

    [dru] : Drum
    [nod] : No Drum

    [cou_fol] : Country folk
    [cla]: Classical
    [pop-roc] : Pop-rock
    [lat-sou] : Latin-soul
    [jaz-blu] : Jazz-blue

    """
    musical = {
        'cel' : 'Cello',
        'cla' : 'Clarinet',
        'flu' : 'Flute',
        'gac' : 'Acoustic guitar',
        'gel' : 'Electric guitar',
        'org' : 'Organ',
        'pia' : 'Piano',
        'sax' : 'Saxophone',
        'tru' : 'Trumpet',
        'vio' : 'Violin',
        'voi' : 'Human Singing Voice'
    }

    info = {
        'dru' : 'Drum',
        'nod' : 'No Drum',
        'cou_fol' : 'Country folk',
        'cla': 'Classical',
        'pop_roc' : 'Pop-rock',
        'lat_sou' : 'Latin-soul',
        'jaz_blu' : 'Jazz-Blue'
    }

    basename = file_name.rstrip('.wav')
    
    pattern = r'\[([^\]]+)\]'
    label = re.findall(pattern, basename)
    output = []

    for i in range(len(label)):
        if i == 0:
            try:
                output.append(musical[label[i]])
            except:
                continue
        else:
            try:
                output.append(info[label[i]])
            except:
                continue
    return output

