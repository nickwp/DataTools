import argparse
import h5py
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='merge hdf5 files with common datasets by concatenating them together')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-o', '--output_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = get_args()
    print("output file:", config.output_file)
    out_file = h5py.File(config.output_file, 'w', track_order=True)
    first_file = h5py.File(config.input_files[0], 'r')
    print(f"opened and checking input file {first_file.filename}")
    keys = first_file.keys()
    attr_keys = first_file.attrs.keys()
    attrs = {k: v if isinstance(v, list) else [v] for k, v in first_file.attrs.items()}
    shapes = {k: list(first_file[k].shape) for k in keys}
    dtypes = {k: first_file[k].dtype for k in keys}
    print(type(first_file.attrs['command']), len(first_file.attrs['command']))
    for fn in config.input_files[1:]:
        f = h5py.File(fn, 'r')
        print(f"opened and checking input file {f.filename}")
        if f.keys() != keys:
            raise KeyError(f"HDF5 file {f.filename} keys {f.keys()} do not match first file's keys {keys}.")
        if f.attrs.keys() != attr_keys:
            raise KeyError(f"HDF5 file {f.filename} attributes {f.attrs.keys()} do not match first file's attributes {attr_keys}.")
        for k in attr_keys:
            attrs[k].extend(f.attrs[k])
        for k in keys:
            shapes[k][0] += f[k].shape[0]
            if shapes[k][1:] != list(f[k].shape[1:]):
                raise ValueError(f"Array {k} in {f.filename} has shape {f[k].shape} which is incompatible with extending previous files shape {shape}.")
    for k in attr_keys:
        print(f"Storing attribute {k}.")
        out_file.attrs[k]  = attrs[k]
    dsets = {k: out_file.create_dataset(k, shape=shapes[k], dtype=dtypes[k]) for k in keys}
    starts = {k: 0 for k in keys}
    offset = 0
    for fn in config.input_files:
        print(f"Opening {fn}")
        f = h5py.File(fn, 'r')
        for k in keys:
            stop = starts[k] + f[k].shape[0]
            isIndex = (k == "event_hits_index")
            print(f"   writing {dtypes[k]} {k} entries {starts[k]} to {stop}" +
                    (f" adding {offset} offest to the values for index array" if isIndex else ""))
            if isIndex:
                dsets[k][starts[k]:stop] = np.array(f[k]) + offset
                offset += f['hit_pmt'].shape[0]
            else:
                dsets[k][starts[k]:stop] = f[k]
            starts[k] = stop
    print(f"Written output file {out_file.filename}.")
    out_file.close()
