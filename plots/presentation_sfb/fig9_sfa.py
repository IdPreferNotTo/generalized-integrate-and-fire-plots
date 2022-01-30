from allensdk.core.nwb_data_set import NwbDataSet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import styles as st

# if you ran the examples above, you will have a NWB file here
home = os.path.expanduser("~")
file_name = home + "/Desktop/human_middle_frontal_gyrus_adap.nwb"
data_set = NwbDataSet(file_name)

sweep_numbers = data_set.get_sweep_numbers()
sweep_number = sweep_numbers[26]
sweep_data = data_set.get_sweep(sweep_number)

# spike times are in seconds relative to the start of the sweep
spike_times = data_set.get_spike_times(sweep_number)

# stimulus is a numpy array in amps
stimulus = sweep_data['stimulus']

# response is a numpy array in volts
response = sweep_data['response']

# sampling rate is in Hz
sampling_rate = sweep_data['sampling_rate']

# start/stop indices that exclude the experimental test pulse (if applicable)
index_range = sweep_data['index_range']

st.set_default_plot_style()
fig = plt.figure(tight_layout=True, figsize=(2.0*3.2, 2.0))
grids = gridspec.GridSpec(1, 1)
ax1 = fig.add_subplot(grids[0])
st.remove_top_right_axis([ax1])

ax1.set_ylabel("$v$ [mV]")
ax1.set_xlabel("$t$ [s]")
ax1.set_xlim([0.75, 1.5])
ax1.plot([0, 1., 1., 2., 2., 3], [-0.05, -0.05, -0.03, -0.03, -0.05, -0.05], c="C3")
ax1.plot([i/sampling_rate for i in range(len(response))], response, c=st.colors[0])
plt.savefig(home + "/Desktop/Presentations/SCC SfB/fig9.png", dpi=300)
plt.show()
