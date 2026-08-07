import h5py

event_num = 0

f = h5py.File("tester.h5", 'r')
print(f)
print(f.keys())
is_truth = f['is_truth'][event_num]
print(f['particle_id'][event_num][~is_truth])
print(f['pt'][event_num][~is_truth])
