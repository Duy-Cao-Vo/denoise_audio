#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
from scipy.linalg import svd


class Denoiser(object):
    '''
    A class for smoothing a noisy, real-valued data sequence by means of SVD of a partial circulant matrix.
    -----
    Attributes:
        mode: str
            Code running mode: "layman" or "expert".
            In the "layman" mode, the code autonomously tries to find the optimal denoised sequence.
            In the "expert" mode, a user has full control over it.
        s: 1D array of floats
            Singular values ordered decreasingly.
        U: 2D array of floats
            A set of left singular vectors as the columns.
        r: int
            Rank of the approximating matrix of the constructed partial circulant matrix from the sequence.
    '''

    def __init__(self, mode="layman", threshold=0.1):
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
        print(X)
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

    def _denoise_for_expert(self, sequence, layer, gap, rank):
        '''
        Smooth a noisy sequence by means of low-rank approximation of its corresponding partial circulant matrix.
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
        print("_denoise_for_expert")
        assert 1 <= rank <= layer <= sequence.size
        self.r = rank
        # linear trend to be deducted
        trend = np.linspace(0, gap, sequence.size)
        X = self._embed(sequence-trend, layer)
        # singular value decomposition
        self.U, self.s, Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # low-rank approximation
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def _cross_validate(self, x, m):
        '''
        Check if the gap of boundary levels of the detrended sequence is within the estimated noise strength.
        -----
        Arguments:
            x: 1D array of floats
                Input array.
            m: int
                Number of rows of the constructed matrix.
        -----
        Returns:
            valid: bool
                Result of cross validation. True means the detrending procedure is valid.
        '''
        X = self._embed(x, m)
        print(X.shape)
        self.U, self.s, self._Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # Search for noise components using the normalized mean total variation of the left singular vectors as an indicator.
        # The procedure runs in batch of every 10 singular vectors.
        self.r = 0
        while True:
            U_sub = self.U[:,self.r:self.r+10]
            NMTV = np.mean(np.abs(np.diff(U_sub,axis=0)), axis=0) / (np.amax(U_sub,axis=0) - np.amin(U_sub,axis=0))
            try:
                # the threshold of 10% can in most cases discriminate noise components
                self.r += np.argwhere(NMTV > self.threshold)[0,0]
                break
            except IndexError:
                self.r += 10
        # estimate the noise strength, while r marks the first noise component
        noise_stdev = np.sqrt(np.sum(self.s[self.r:]**2) / X.size)
        # estimate the gap of boundary levels after detrend
        gap = np.abs(x[-self._k:].mean()-x[:self._k].mean())
        valid = gap < noise_stdev
        return valid

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
        print("_denoise_for_layman")
        assert 1 <= layer <= sequence.size
        # The code takes the mean of a few neighboring data to estimate the boundary levels of the sequence.
        # By default, this number is 11.
        self._k = 11
        print(self._k)
        # Initially, the code assumes no linear inclination.
        trend = np.zeros_like(sequence)
        # Iterate over the averaging length.
        # In the worst case, iteration must terminate when it is 1.
        while not self._cross_validate(sequence-trend, layer):
            self._k -= 2
            trend = np.linspace(0, sequence[-self._k:].mean()-sequence[:self._k].mean(), sequence.size)
        # low-rank approximation by using only signal components
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        print(self._k)
        return denoised

    def denoise(self, *args, **kwargs):
        '''
        User interface method.
        It will reference to different denoising methods ad hoc under the fixed name.
        '''
        print(self.mode)
        return self._method[self.mode](*args, **kwargs)

from pydub import AudioSegment
import pydub
import librosa

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
#     denoiser_pca = Denoiser()
#     denoised = denoiser_pca.denoise(sequence, 500)

#     write("la_lung_clean_svd.wav", sr, denoised, True)

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


import pandas as pd
import os
noise_folder = "D:/Cubase/output/duy_noise/"
raw_folder =  "D:/Cubase/output/duy_raw/"
threshold = 0.05
data = []
method = 'svd'

for f in sorted(os.listdir(noise_folder))[:]:
    if f.endswith(('wav', 'mp3')):
        noise_path = os.path.join(noise_folder, f)
        print(noise_path)
        sequence, sr = librosa.load(noise_path, sr=None)
        # Define the duration to slice (1 minute)
        duration = 1 * 10 * sr + 5* sr # 1 minute in milliseconds

        # Slice the audio to the first minute
        sequence = sequence[5* sr:duration]
        denoiser = Denoiser(threshold=threshold)
        denoised = denoiser.denoise(sequence, 500)
        
        denoised_path = os.path.join(raw_folder, f.replace('noise', 'raw'))
        raw, sr = librosa.load(denoised_path, sr=None)
        raw = raw[5* sr:duration]

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
