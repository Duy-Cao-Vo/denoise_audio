import os
import random
from pydub import AudioSegment
import pydub
import numpy as np

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_file(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

    
def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
import numpy as np

clean_folder =  "D:\\master yi\\Untitled Folder\\daps\\clean"
noise_folder = "noise"
data = []



clean_files=[os.path.join(clean_folder, f) for f in os.listdir(clean_folder)]
clean_files.sort()
    
noise_files=[os.path.join(noise_folder, f) for f in os.listdir(noise_folder)]
noises={}

for nf in noise_files:
    fr, n_sgn=read(nf)
    print(f"{n_sgn.shape} {fr} {nf}")
    
    noises[os.path.basename(nf).split('.')[0]] = n_sgn
    # n_sgn=np.sum(n_sgn, 1)/2
    
for nf in noises:
    for cf in sorted(clean_files[:20]):
        if "wav" in cf:
            fr, c_sgn = read(cf)
            # Read and preprocess the noise signal
            s_noi = noises[nf]
            s_noi = s_noi * 0.2
            s_noi = s_noi.astype(int)

            # Get the first 5 seconds of s_noi and duplicate it to make it 10 seconds
            first_5s_length = 3 * fr
            s_noi_5s = s_noi[:first_5s_length]
            s_noi_10s = np.tile(s_noi_5s, 4)  # Duplicate to create 10s

            # Trim the clean signal to match the length of the 10-second noise signal
            c_sgn = c_sgn[5 * fr:len(s_noi_10s) + 5 * fr]

            # Add the noise to the clean signal
            n_sgn = c_sgn.copy()
            n_sgn = n_sgn + s_noi_10s

            mkdir_if_not_exist(f"clean_{nf}")
            write(f"clean_{nf}/{os.path.basename(cf)}", fr, c_sgn)
            mkdir_if_not_exist(f"noise_{nf}")
            write(f"noise_{nf}/{os.path.basename(cf)}", fr, n_sgn)

            print(fr)
