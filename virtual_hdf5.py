import argparse
import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create virtual dataset')
    parser.add_argument('files', type=str, nargs='+', help='path to input h5 files')
    parser.add_argument('-o', '--out', type=str, default='./out.h5',
                        help='output virtual file.')
    args = parser.parse_args()

    files = [h5py.File(ff, 'r') for ff in args.files]

    ids = np.concatenate([ff['exam_id'][:-1] for ff in files])
    #regs = np.concatenate([ff['register_num'] for ff in files])

    f = h5py.File(args.out, 'w')

    f.create_dataset('exam_id', data=ids, dtype='i8')
    #f.create_dataset('register_num', data=ids, dtype='i4')

    layout = h5py.VirtualLayout(shape=(len(ids), 4096, 12), dtype='f4')

    end = 0
    for i, file in enumerate(args.files):
        start = end
        end = start + len(files[i]['exam_id'][:-1])
        print(start, end)
        vsource = h5py.VirtualSource(file, 'tracings', shape=(end-start, 4096, 12))
        layout[start:end, :, :] = vsource

    f.create_virtual_dataset('tracings', layout, fillvalue=0)
    f.close()