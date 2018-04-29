import scipy.signal as signal
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import adaptfilt

FREQ_INC_AMT = .01 # Hz

def notch_filter(input_signal, sampling_period, freq=60, r=.99):
  print freq
  N = np.cos(2*np.pi*freq*sampling_period)
  # Equation from textbook
  b = [1, -2*N, 1]
  a = [1., -2.*r*N, r**2]
  b_0 = sum(a)/sum(b)

  b = np.array(b) * b_0
  return signal.lfilter(b, a, input_signal)


def band_pass(input_signal, w_lo=, w_hi):
  w_lo = 55./500
  w_hi = 65./500
  b, a = signal.butter(1, [w_lo, w_hi], "bandpass")
  
  return signal.lfilter(b, a, input_signal)

"""
adaptive_notch - applies the notch filter and perterbs the frequency to try to
    adapt to the input signal. 
parameters:
  input_signal: Array containing the values of the input signal.
  sampling_period: Float representing the time interval between samples (in
    seconds)
  start_freq: The frequency the notch filter will start at - a best guess of
    where the noise will be.
  r: The radius of the zero from the origin (controls the steepness of the
    notch filter).
  b_0: Normalizing constant.
  time_to_adapt: Time in seconds allowed for the filter to come to equilibrium
    each time the notch freqency is perterbed.
returns:
  An array containing the filtered input signal.
"""
def adaptive_notch(input_signal, sampling_period=.001, start_freq=60, r=.99,
                   samples_to_adapt=600, double_filter=False):
  # Calculate the number of samples in each segment.
  samples_per_segment = 3*samples_to_adapt
  samples_btwn_segments = 2*samples_to_adapt

  freq = start_freq
  next_segment = []
  output = []
  next_segment[:] = input_signal[0:samples_per_segment] 
  filtered_segment = notch_filter(next_segment, sampling_period, freq, r)
  double_filtered_segment = band_pass(filtered_segment)
  
  output = np.concatenate([output, filtered_segment])

  prev_e = energy(double_filtered_segment[samples_to_adapt:])
  last_inc_dir = 1
  i = samples_btwn_segments
  energies = []
  while i < (len(input_signal) - samples_to_adapt):
    end = i + samples_per_segment
    if end > len(input_signal):
      end = len(input_signal)
    next_segment[:] = input_signal[i:end]
    filtered_segment = notch_filter(next_segment, sampling_period, freq, r)
    if double_filter:
      doubled_filtered_segment = band_pass(filtered_segment)
      e = energy(double_filtered_segment[samples_to_adapt:])
    else:
      e = energy(filtered_segment[samples_to_adapt:])
    print "energy", e
    energies.append(e)
    if e < prev_e:
      # Move frequency in the same direction as last time.
      freq += last_inc_dir*FREQ_INC_AMT 
    else: 
      # Move frequency in the opposite direction from last time.
      last_inc_dir *= -1
      freq += last_inc_dir*FREQ_INC_AMT
    output = np.concatenate([output, filtered_segment[samples_to_adapt:]])
    prev_e = e
    i += samples_btwn_segments

  return output, energy

def energy(signal):
  return sum(np.power(signal, 2))


def mse(signal_a, signal_b):
  return ((np.array(signal_a) - np.array(signal_b)) ** 2).mean()

if __name__=="__main__":
  if len(sys.argv) < 3: 
    print(
      "please give commandline argument [path to signal with noise]" +
      " [path to signal without noise]")
    exit()
  noise_file = sys.argv[1]
  no_noise_file = sys.argv[2]
  if len(sys.argv) >= 3:
    # optional commandline arg determines if energy is based on band-passed
    # filtered signal (true) or just the filtered signal (false)
    double_filter = bool(sys.argv[2])
  
  sampling_period = .001
  
  noise_data = []
  with open(noise_file) as noise_csv:
    reader = csv.reader(noise_csv)
    for row in reader:
      noise_data.append(float(row[0]))

  no_noise_signal = []
  with open(no_noise_file) as no_noise_csv:
    reader = csv.reader(no_noise_csv)
    # Skip the first two rows - header info.
    next(reader)
    next(reader)
    for row in reader:
      no_noise_signal.append(float(row[1]))

  filtered = notch_filter(noise_data, sampling_period)

  adapt_filtered = adaptive_notch(noise_data)
  """
  with open("output_4-28-001.csv", "wb") as adapt_output:
    writer = csv.writer(adapt_output)
    for item in adapt_filtered:
      writer.writerow([item])
  """

  # calculate the mean squared error. 
  mse_std_notch = mse(no_noise_signal, filtered)
  mse_adapt_notch = mse(no_noise_signal, adapt_filtered)
  print "standard notch error:", mse_std_notch
  print "adaptive notch error:", mse_adapt_notch

  # plot the original unfiltered noise data
  plt.figure()
  plt.plot(noise_data)
  plt.title('unfiltered')
  
  plt.figure()
  plt.plot(filtered)
  plt.title('standard notch filter output')
  print filtered

  plt.figure()
  plt.plot(adapt_filtered)
  plt.title('adaptive notch filter output')

  plt.figure()
  plt.plot(band_pass(noise_data[:5000]))
  plt.title("ahhhhhhhhhhhhhhhhhhhh")

  plt.show()
