import h5py

path = "sift-128-euclidean.hdf5"

with h5py.File(path, "r") as f:
    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)

    f.visititems(inspect)