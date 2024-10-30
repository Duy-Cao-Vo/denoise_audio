import numpy as np
from sklearn.decomposition import PCA
import librosa
import pydub

class DenoiserPCA(object):
    '''
    A class for smoothing a noisy, real-valued data sequence by means of PCA of a partial circulant matrix.
    -----
    Attributes:
        mode: str
            Code running mode: "layman" or "expert".
            In the "layman" mode, the code autonomously tries to find the optimal denoised sequence.
            In the "expert" mode, a user has full control over it.
        explained_variance_ratio_: 1D array of floats
            Explained variance ratios for each principal component.
        components_: 2D array of floats
            Principal components.
        r: int
            Number of principal components selected.
    '''

    def __init__(self, mode="layman", threshold=0.05):
        '''
        Class initialization.
        -----
        Arguments:
            mode: str
                Denoising mode. To be selected from ["layman", "expert"]. Default is "layman".
                While "layman" grants the code autonomy, "expert" allows a user to experiment.
        -----
        Raises:
            ValueError
                If mode is neither "layman" nor "expert".
        '''
        self._method = {"layman": self._denoise_for_layman, "expert": self._denoise_for_expert}
        if mode not in self._method:
            raise ValueError("unknown mode '{:s}'!".format(mode))
        self.mode = mode
        self.threshold = threshold

    def _embed(self, x, m):
        '''
        Embed a 1D array into a 2D partial circulant matrix by cyclic left-shift.
        -----
        Arguments:
            x: 1D array of floats
                Input array.
            m: int
                Number of rows of the constructed matrix.
        -----
        Returns:
            X: 2D array of floats
                Constructed partial circulant matrix.
        '''
        x_ext = np.hstack((x, x[:m-1]))
        shape = (m, x.size)
        strides = (x_ext.strides[0], x_ext.strides[0])
        X = np.lib.stride_tricks.as_strided(x_ext, shape, strides)
        return X

    def _reduce(self, A):
        '''
        Reduce a 2D matrix to a 1D array by cyclic anti-diagonal average.
        -----
        Arguments:
            A: 2D array of floats
                Input matrix.
        -----
        Returns:
            a: 1D array of floats
                Output array.
        '''
        m = A.shape[0]
        A_ext = np.hstack((A[:,-m+1:], A))
        strides = (A_ext.strides[0]-A_ext.strides[1], A_ext.strides[1])
        a = np.mean(np.lib.stride_tricks.as_strided(A_ext[:,m-1:], A.shape, strides), axis=0)
        return a

    def _determine_optimal_components(self, explained_variance_ratio):
        '''
        Determine the optimal number of components based on VRE (Variance Retained Explanation).
        -----
        Arguments:
            explained_variance_ratio: 1D array of floats
                The ratio of variance explained by each principal component.
            threshold: float
                The minimum cumulative variance explained required to determine the optimal number of components.
        -----
        Returns:
            r: int
                Optimal number of components.
        '''
        cumulative_variance = np.cumsum(explained_variance_ratio)
        r = np.argmax(cumulative_variance >= (1-self.threshold)) + 1
        return r

    def _denoise_for_expert(self, sequence, layer, gap, rank):
        '''
        Smooth a noisy sequence using PCA and low-rank approximation.
        -----
        Arguments:
            sequence: 1D array of floats
                Data sequence to be denoised.
            layer: int
                Number of leading rows selected from the matrix.
            gap: float
                Gap between the data levels on the left and right ends of the sequence.
                A positive value means the right level is higher.
            rank: int
                Rank of the approximating matrix.
        -----
        Returns:
            denoised: 1D array of floats
                Smoothed sequence after denoise.
        -----
        Raises:
            AssertionError
                If condition 1 <= rank <= layer <= sequence.size cannot be fulfilled.
        '''
        assert 1 <= rank <= layer <= sequence.size
        self.r = rank
        trend = np.linspace(0, gap, sequence.size)
        X = self._embed(sequence - trend, layer)
        
        # PCA decomposition
        pca = PCA(n_components=self.r)
        X_pca = pca.fit_transform(X)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.components_ = pca.components_
        
        # Reconstruct the low-rank approximation
        A = pca.inverse_transform(X_pca)
        denoised = self._reduce(A) + trend
        return denoised

    def _denoise_for_layman(self, sequence, layer):
        '''
        Similar to the "expert" method, except that denoising parameters are optimized autonomously.
        -----
        Arguments:
            sequence: 1D array of floats
                Data sequence to be denoised.
            layer: int
                Number of leading rows selected from the corresponding circulant matrix.
        -----
        Returns:
            denoised: 1D array of floats
                Smoothed sequence after denoise.
        -----
        Raises:
            AssertionError
                If condition 1 <= layer <= sequence.size cannot be fulfilled.
        '''
        assert 1 <= layer <= sequence.size
        self._k = 11
        trend = np.zeros_like(sequence)
        X = self._embed(sequence - trend, layer)
        
        pca = PCA()
        X_pca = pca.fit_transform(X)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        # Determine the optimal number of components
        self.r = self._determine_optimal_components(self.explained_variance_ratio_)
        print("DEBUG self.r", self.r)
        pca = PCA(n_components=self.r)
        X_pca = pca.fit_transform(X)
        A = pca.inverse_transform(X_pca)

        denoised = self._reduce(A) + trend
        return denoised

    def denoise(self, *args, **kwargs):
        '''
        User interface method.
        It will reference different denoising methods ad hoc under the fixed name.
        '''
        return self._method[self.mode](*args, **kwargs)
    
    
def write(f, sr, x, normalized=False):
    """Converts a numpy array to a WAV file."""
    # Validate that x is a 1D or 2D numpy array
    if x.ndim not in [1, 2]:
        raise ValueError("Input array must be 1D or 2D representing mono or stereo audio.")
    
    # Determine the number of channels
    channels = 1 if x.ndim == 1 else x.shape[1]
    
    # Normalize if required
    if normalized:
        # Clip values to [-1, 1] to prevent distortion
        x = np.clip(x, -1, 1)
        y = (x * 32767).astype(np.int16)
    else:
        # Assume x is already in the range of int16
        y = x.astype(np.int16)
    
    # Create an audio segment using pydub
    song = pydub.AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit audio
        channels=channels
    )
    song.export(f, format="wav", bitrate="44100")
    
    
# Test usage
# if __name__ == "__main__":
#     from pydub import AudioSegment
#     import pydub
#     noise_path = 'D:/Cubase/output/la_lung_noise.mp3'
#     sequence, sr = librosa.load(noise_path, sr=None)
#     denoiser_pca = DenoiserPCA()
#     denoised = denoiser_pca.denoise(sequence, 500)

#     write("la_lung_clean_pca.wav", sr, denoised, True)
import numpy as np

"""
inputs
    clean_signal: signal without noise, in shape of 1D
    denoised_signal: signal after denoised, in shape of 1D
"""
def nomalize(x):
    x = np.clip(x, -1, 1)
    y = (x * 32767).astype(np.int16)
    return y

def SNR(clean_signal, denoised_signal):
    # Normalize both signals
#     clean_signal = clean_signal
#     denoised_signal = nomalize(denoised_signal)
    
    # Calculate the power of the denoised signal
    dns_a = np.sum(np.power(denoised_signal, 2))
    
    # Calculate the noise signal (difference between clean and denoised)
    noise_signal = clean_signal - denoised_signal
    nos_a = np.sum(np.power(noise_signal, 2))

    # Print debug information
    print(f"Power of denoised signal: {dns_a}")
    print(f"Power of noise signal: {nos_a}")

    # Avoid division by zero
    if nos_a == 0:
        return float('inf')  # Infinite SNR if noise power is zero

    # Calculate SNR in dB
    snr = 20 * np.log10(np.sqrt(dns_a) / np.sqrt(nos_a))
    
    return snr


def MSE(clean_signal, denoised_signal):
    mse=(1/len(clean_signal))*np.sum(np.power(np.subtract(clean_signal, denoised_signal), 2))
    return mse

import numpy as np
import librosa
from scipy.signal import butter, sosfilt, correlate

def bandpass_filter(signal, sr, lowcut=300, highcut=3400, order=5):
    """Apply an optimized bandpass filter using SOS."""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal)

def time_align(clean, degraded, sr):
    """Efficient time alignment using downsampled cross-correlation."""
    downsample_factor = 10
    clean_ds = librosa.resample(clean, orig_sr=sr, target_sr=sr // downsample_factor)
    degraded_ds = librosa.resample(degraded, orig_sr=sr, target_sr=sr // downsample_factor)
    
    correlation = correlate(degraded_ds, clean_ds, mode='full')
    delay = np.argmax(correlation) - len(clean_ds) + 1
    if delay > 0:
        aligned_degraded = np.pad(degraded, (delay * downsample_factor, 0), 'constant')[:len(clean)]
    else:
        aligned_degraded = np.pad(degraded, (0, -delay * downsample_factor), 'constant')[:len(clean)]
    return aligned_degraded

def bark_scale_transform(signal, sr):
    """Vectorized Bark scale transformation."""
    n = len(signal)
    # Pad to the nearest power of 2 for faster FFT
    n_padded = 2**np.ceil(np.log2(n)).astype(int)
    signal = np.pad(signal, (0, n_padded - n), 'constant')
    
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    
    bark_edges = np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1175, 
                           1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 
                           5800, 7000, 8500, 10500, 13500])
    
    # Vectorized calculation of Bark bands
    bark_bands = np.array([np.mean(np.abs(fft[(freqs >= bark_edges[i]) & (freqs < bark_edges[i+1])]))
                           for i in range(len(bark_edges)-1)])
    return bark_bands

def pesq(clean, degraded, sr):
    """Optimized PESQ-like score."""
    # Step 1: Bandpass filter
    clean_filtered = bandpass_filter(clean, sr)
    degraded_filtered = bandpass_filter(degraded, sr)
    
    # Step 2: Time alignment
    aligned_degraded = time_align(clean_filtered, degraded_filtered, sr)
    
    # Step 3: Perceptual modeling (Bark scale transform)
    clean_bark = bark_scale_transform(clean_filtered, sr)
    degraded_bark = bark_scale_transform(aligned_degraded, sr)
    
    # Step 4: Distortion calculation
    symmetrical_distortion = np.mean(np.abs(clean_bark - degraded_bark))
    
    print(symmetrical_distortion)
    
    # Convert distortion to a PESQ-like score (simplified mapping)
    pesq_score = max(0, 4.5 - symmetrical_distortion)  # 4.5 is the max PESQ score
    
    return symmetrical_distortion


### RAIN ###
import pandas as pd
import os
noise_folder = "noise_rain/"
raw_folder =  "clean_rain/"
data = []
method = 'pca'
for threshold in np.arange(0.001, 0.01, 0.002):
    threshold = round(threshold,3)
    print(threshold)
    for f in sorted(os.listdir(noise_folder))[:20]:
        if f.endswith(('wav', 'mp3')):
            noise_path = os.path.join(noise_folder, f)
            print(noise_path)
            sequence, sr = librosa.load(noise_path, sr=None)
            # Define the duration to slice (1 minute)
            duration = 2 * 10 * sr # 10s in milliseconds

            # Slice the audio to the first minute
            sequence = sequence[:duration]
            denoiser = DenoiserPCA(threshold=threshold)
            denoised = denoiser.denoise(sequence, 200)

            denoised_path = os.path.join(raw_folder, f)
            raw, sr = librosa.load(denoised_path, sr=None)
            raw = raw[:duration]

            write(f"./{method}_denoised/{f}", sr, denoised, True)
            write(f"./{method}_noise/{f}", sr, sequence, True)
            write(f"./{method}_raw/{f}", sr, raw, True)

            denoised, sr_A = librosa.load(f"./{method}_denoised/{f}", sr=None)
            raw, sr_A = librosa.load(f"./{method}_raw/{f}", sr=None)


            snr = SNR(raw, denoised)


            # Compute the FFT of each signal
            from scipy.fft import fft, fftfreq
            denoised_fft = fft(denoised)
            raw_fft = fft(raw)

            denoised_fft_magnitude = np.abs(denoised_fft[:len(denoised_fft)//2])
            raw_fft_magnitude = np.abs(raw_fft[:len(raw_fft)//2])
            from scipy.spatial.distance import cosine
            cos_sim = 1 - cosine(denoised_fft_magnitude, raw_fft_magnitude)

            correlation = np.corrcoef(denoised, raw)[0, 1]

            pesq_score = pesq(denoised, raw, sr_A)

            data.append({"file_raw": f"./{method}_raw/{f}", 
                         "file_denoised": f"./{method}_denoised/{f}", 
                         "method": method, 
                         "threshold": threshold, 
                         "SNR": snr, 'pesq_score': pesq_score, "cos Spectrum": cos_sim, 'correlation': correlation})
    df =  pd.DataFrame(data)
    df.to_csv(f"{method}_{str(threshold).replace('.','')}_rain.csv", index=False)


### TRAFFIC ###
import pandas as pd
import os
noise_folder = "noise_traffic/"
raw_folder =  "clean_traffic/"
data = []
method = 'pca'
for threshold in [0.01]:
    threshold = round(threshold,3)
    print(threshold)
    for f in sorted(os.listdir(noise_folder))[:20]:
        if f.endswith(('wav', 'mp3')):
            noise_path = os.path.join(noise_folder, f)
            print(noise_path)
            sequence, sr = librosa.load(noise_path, sr=None)
            # Define the duration to slice (1 minute)
            duration = 2 * 10 * sr # 10s in milliseconds

            # Slice the audio to the first minute
            sequence = sequence[:duration]
            denoiser = DenoiserPCA(threshold=threshold)
            denoised = denoiser.denoise(sequence, 200)

            denoised_path = os.path.join(raw_folder, f)
            raw, sr = librosa.load(denoised_path, sr=None)
            raw = raw[:duration]

            write(f"./{method}_denoised/{f}", sr, denoised, True)
            write(f"./{method}_noise/{f}", sr, sequence, True)
            write(f"./{method}_raw/{f}", sr, raw, True)

            denoised, sr_A = librosa.load(f"./{method}_denoised/{f}", sr=None)
            raw, sr_A = librosa.load(f"./{method}_raw/{f}", sr=None)


            snr = SNR(raw, denoised)


            # Compute the FFT of each signal
            from scipy.fft import fft, fftfreq
            denoised_fft = fft(denoised)
            raw_fft = fft(raw)

            denoised_fft_magnitude = np.abs(denoised_fft[:len(denoised_fft)//2])
            raw_fft_magnitude = np.abs(raw_fft[:len(raw_fft)//2])
            from scipy.spatial.distance import cosine
            cos_sim = 1 - cosine(denoised_fft_magnitude, raw_fft_magnitude)

            correlation = np.corrcoef(denoised, raw)[0, 1]

            pesq_score = pesq(denoised, raw, sr_A)

            data.append({"file_raw": f"./{method}_raw/{f}", 
                         "file_denoised": f"./{method}_denoised/{f}", 
                         "method": method, 
                         "threshold": threshold, 
                         "SNR": snr, 'pesq_score': pesq_score, "cos Spectrum": cos_sim, 'correlation': correlation})
    df =  pd.DataFrame(data)
    df.to_csv(f"{method}_{str(threshold).replace('.','')}_traffic.csv", index=False)

### WORKING PLACE ###
import pandas as pd
import os
noise_folder = "noise_working place//"
raw_folder =  "clean_working place/"
data = []
method = 'pca'
for threshold in [0.01]:
    threshold = round(threshold,3)
    print(threshold)
    for f in sorted(os.listdir(noise_folder))[:20]:
        if f.endswith(('wav', 'mp3')):
            noise_path = os.path.join(noise_folder, f)
            print(noise_path)
            sequence, sr = librosa.load(noise_path, sr=None)
            # Define the duration to slice (1 minute)
            duration = 2 * 10 * sr # 10s in milliseconds

            # Slice the audio to the first minute
            sequence = sequence[:duration]
            denoiser = DenoiserPCA(threshold=threshold)
            denoised = denoiser.denoise(sequence, 200)

            denoised_path = os.path.join(raw_folder, f)
            raw, sr = librosa.load(denoised_path, sr=None)
            raw = raw[:duration]

            write(f"./{method}_denoised/{f}", sr, denoised, True)
            write(f"./{method}_noise/{f}", sr, sequence, True)
            write(f"./{method}_raw/{f}", sr, raw, True)

            denoised, sr_A = librosa.load(f"./{method}_denoised/{f}", sr=None)
            raw, sr_A = librosa.load(f"./{method}_raw/{f}", sr=None)


            snr = SNR(raw, denoised)


            # Compute the FFT of each signal
            from scipy.fft import fft, fftfreq
            denoised_fft = fft(denoised)
            raw_fft = fft(raw)

            denoised_fft_magnitude = np.abs(denoised_fft[:len(denoised_fft)//2])
            raw_fft_magnitude = np.abs(raw_fft[:len(raw_fft)//2])
            from scipy.spatial.distance import cosine
            cos_sim = 1 - cosine(denoised_fft_magnitude, raw_fft_magnitude)

            correlation = np.corrcoef(denoised, raw)[0, 1]

            pesq_score = pesq(denoised, raw, sr_A)

            data.append({"file_raw": f"./{method}_raw/{f}", 
                         "file_denoised": f"./{method}_denoised/{f}", 
                         "method": method, 
                         "threshold": threshold, 
                         "SNR": snr, 'pesq_score': pesq_score, "cos Spectrum": cos_sim, 'correlation': correlation})
    df =  pd.DataFrame(data)
    df.to_csv(f"{method}_{str(threshold).replace('.','')}_working place.csv", index=False)
