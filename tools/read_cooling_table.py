import h5py

with h5py.File("chianti_cooling_table.h5", "r") as f:
    temperature = f["temperature_K"][:]
    electron_density = f["electron_density_cm-3"][:]
    metallicity = f["metallicity_Zsun"][:]
    cooling = f["cooling_erg_cm3_s"][:]

print(cooling.shape)
# Shape is:
#   nZ, nT, nne
