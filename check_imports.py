import platform
import numpy as np
import faiss
import h5py

print("Platform:", platform.platform())
print("Processor:", platform.processor())
print("NumPy:", np.__version__)
print("FAISS:", faiss.__version__)
print("h5py:", h5py.__version__)
print("FAISS threads:", faiss.omp_get_max_threads())