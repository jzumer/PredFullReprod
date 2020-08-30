import tensorflow.keras as k
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Flatten, Activation, BatchNormalization, LayerNormalization
from tensorflow.keras import Model, Input

import keras.utils.generic_utils as ku

from PredFull.coord_tf import CoordinateChannel2D, CoordinateChannel1D

import numpy
import tables
import time
import json
import pickle

import os
import sys
import itertools

import time

USE_PRETRAINED = False # Use pretrained model instead of our reproduction thereof
RESET_PRETRAINED = True # Reinit the weights of the pretrained model (no effect when USE_PRETRAINED is False)
PRETRAINED = "pm.h5" # Pretrained model to use, only relevant if USE_PRETRAINED = True
DATA = "ProteomeTools.h5" # Dataset to use, as prepared by `predfull_data_to_h5.py`

REPROD = False
PHASE = None
LR = None
LOAD = None
SAVE = None
if len(sys.argv) > 1:
    PHASE = int(sys.argv[1])
    REPROD = True

    FMT = "pm_re{}.h5" if not USE_PRETRAINED else "pm_over{}.h5"
    SAVE = FMT.format(PHASE)
    LOAD = FMT.format(PHASE-1)

    if PHASE == 0:
        LOAD = None if not USE_PRETRAINED else PRETRAINED
        LR = 3e-4
        N_EPS = 30
    elif PHASE == 1:
        LR = 5e-5
        N_EPS = 10
    elif PHASE == 2:
        LR = 1.25e-5
        N_EPS = 10

PROT_BLIT_LEN = 2000 * 10
BATCH_SIZE = 512
PROT_STR_LEN = 30

# This part copied from the PredFull repo
Alist = list('ACDEFGHIKLMNPQRSTVWY')
oh_dim = len(Alist) + 2

charMap = {'@': 0, '[': 21}
for i, a in enumerate(Alist):
    charMap[a] = i + 1

mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}
# End of copied segment

raw_data = tables.open_file(DATA, 'r')

SEQ_DIM = (len(AMINOS) - 2) + 2 + 2 # mod is still there, though set to 0 in this reproduction (because this is a new feature not in the original paper)

def cos_metric(y_true, y_pred):
    y1 = K.l2_normalize(y_true, axis=-1)
    x1 = K.l2_normalize(y_pred, axis=-1)
    cos = K.sum(y1 * x1, axis=-1)
    return cos

def cos_loss(y_true, y_pred):
    return -cos_metric(y_true, y_pred)

# Squeeze-and-Excitation blocks
def se(inp):
    se_shape = (1, inp.shape[-1])

    out = k.layers.Conv1D(se_shape[1], 3, activation=None, padding='same')(inp)
    out_pre = k.layers.BatchNormalization()(out)
    out = k.layers.GlobalAveragePooling1D()(out_pre)
    out = k.layers.Dense(se_shape[1], activation=None)(out)
    out = k.layers.Reshape(se_shape)(out)

    out = k.layers.multiply([out_pre, out])
    out = k.layers.add([inp, out])
    out = k.layers.Activation('relu')(out)

    return out

# Residual blocks
def res(inp):
    out = k.layers.Conv1D(inp.shape[-1], 1, activation=None, padding='same')(inp)
    out = k.layers.BatchNormalization()(out)
    out = k.layers.add([out, inp])
    out = k.layers.Activation('relu')(out)

    return out

def make_model():
    inp = k.layers.Input(shape=(32, 24))

    out = inp
    out = CoordinateChannel1D()(inp)

    hid_size = 64

    out_whole = k.layers.Conv1D(hid_size * 8, 1, padding='same', activation=None)(out)
    out_whole = k.layers.BatchNormalization()(out_whole)
    
    out2 = k.layers.Conv1D(hid_size, 2, padding='same', activation=None)(out)
    out2 = k.layers.BatchNormalization()(out2)
    out3 = k.layers.Conv1D(hid_size, 3, padding='same', activation=None)(out)
    out3 = k.layers.BatchNormalization()(out3)
    out4 = k.layers.Conv1D(hid_size, 4, padding='same', activation=None)(out)
    out4 = k.layers.BatchNormalization()(out4)
    out5 = k.layers.Conv1D(hid_size, 5, padding='same', activation=None)(out)
    out5 = k.layers.BatchNormalization()(out5)
    out6 = k.layers.Conv1D(hid_size, 6, padding='same', activation=None)(out)
    out6 = k.layers.BatchNormalization()(out6)
    out7 = k.layers.Conv1D(hid_size, 7, padding='same', activation=None)(out)
    out7 = k.layers.BatchNormalization()(out7)
    out8 = k.layers.Conv1D(hid_size, 8, padding='same', activation=None)(out)
    out8 = k.layers.BatchNormalization()(out8)
    out9 = k.layers.Conv1D(hid_size, 9, padding='same', activation=None)(out)
    out9 = k.layers.BatchNormalization()(out9)

    out = k.layers.concatenate([out2, out3, out4, out5, out6, out7, out8, out9])
    out = k.layers.add([out, out_whole])
    out = k.layers.Activation('relu')(out)

    out = se(out)
    out = se(out)
    out = se(out)
    out = se(out)
    out = se(out)
    out = se(out)
    out = se(out)
    out = se(out)

    out = res(out)
    out = res(out)
    out = res(out)

    out = k.layers.Conv1D(2000*10, 1, activation=None, padding='same')(out)
    out = k.layers.Activation('sigmoid')(out)
    out = k.layers.GlobalAveragePooling1D()(out)

    return inp, out

K.clear_session()

ku.get_custom_objects().update({"cos_metric": cos_metric})

if REPROD and (LOAD is not None):
    pm2 = k.models.load_model(LOAD, compile=0)
    if USE_PRETRAINED and RESET_PRETRAINED:
        for layer in pm2.layers:
            params = []
            if hasattr(layer, 'kernel_initializer'): 
                w = layer.kernel_initializer(layer.weights[0].shape)
                params.append(w)
            if hasattr(layer, 'bias_initializer'):
                b = layer.bias_initializer(layer.weights[1].shape)
                params.append(b)
            if len(params) > 0:
                layer.set_weights(params)
else:
    ipt, opt = make_model()
    pm2 = k.models.Model(ipt, opt)

pm2.compile(optimizer=k.optimizers.Adam(lr=LR if REPROD else 1e-3), loss=cos_loss, metrics=[cos_metric])

pm2.summary()
print()
print(f"Parameters for this run: phase = {PHASE}; Using pretrained model: {USE_PRETRAINED}; Reset pretrained params: {RESET_PRETRAINED}; learning rate: {LR}; Load filename: {LOAD}; Save filename: {SAVE}; Dataset: {DATA}")

def blit(mz_list, itensity_list, mass, bin_size, charge): #spectra2vector as copied from the PredFull repo
    itensity_list = itensity_list / numpy.max(itensity_list)
    vector = numpy.zeros(PROT_BLIT_LEN, dtype='float32')
    mz_list = numpy.asarray(mz_list)
    indexes = mz_list / bin_size
    indexes = numpy.around(indexes).astype('int32')
    for i, index in enumerate(indexes):
        if index < len(vector):
            vector[index] += itensity_list[i]

    # normalize
    vector = numpy.sqrt(vector)

    # remove precursors, including isotropic peaks
    for delta in (0, 1, 2):
        precursor_mz = mass + delta / charge
        if precursor_mz > 0 and precursor_mz < ((PROT_BLIT_LEN / bin_size) - 0.5):
            vector[int(round(precursor_mz / bin_size))] = 0

    return vector

def embed(sp, mass_scale = 2000, out=None, ignore=False, pep=None): # As copied from the PredFull repo
# Input is {"pep": peptide, "charge": charge, "mass": precursor mass, "type": 3 = 'hcd', "nce": 25"}
    if out is None: em = numpy.zeros((PROT_STR_LEN + 2, SEQ_DIM), dtype='float32')
    else: em = out

    if pep is None: pep = sp['pep']

    if len(pep) > PROT_STR_LEN and ignore != False: return em # too long

    em[len(pep)][21] = 1 # ending pos, next line with +1 to skip this
    for i in range(len(pep) + 1, PROT_STR_LEN + 1): em[i][0] = 1 # padding first, meta column should not be affected
    meta = em[-1]
    meta[0] = (sp['mass'] * sp['charge'] - (sp['charge'] - 1)) / mass_scale # pos 0, and overwrtie above padding
    meta[sp['charge']] = 1 # pos 1 - 4
    meta[5 + sp['type']] = 1 # pos 5 - 8
    meta[-1] = sp['nce'] / 10000.0 if 'nce' in sp else 0.0025
    for i in range(len(pep)):
        em[i][charMap[pep[i]]] = 1 # 1 - 20
        em[i][-1] = mono[pep[i]] / mass_scale

    return em              

class MsDataset():
    def __init__(self, data, idxs, max_charge=7, epoch_len=None, theoretical=True, generated=False, shuffle=False):
        self.data = data
        self.idxs = idxs
        numpy.random.shuffle(self.idxs)
        self.theoretical = theoretical or generated
        self.generated = generated
        self.shuffle = shuffle
        self.len = epoch_len if epoch_len is not None else len(idxs)
        self.hitlist = {}
        self.max_charge = max_charge

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = numpy.random.choice(self.len)
        charge = self.data.root.meta[self.idxs[idx]]['charge']
        mass = self.data.root.meta[self.idxs[idx]]['mass']
        seq = self.data.root.meta[self.idxs[idx]]['seq'].decode("utf-8")
        spec = self.data.root.spectrum[self.idxs[idx]]

        seq_spec = {"pep": seq, "charge": charge, "mass": mass, "type": 3, "nce": 25}
        
        enc_seq = numpy.array(embed(seq_spec))

        out_set = []
        out_set.append(enc_seq.reshape((1, *enc_seq.shape)))
        out_set.append(spec.reshape((1, *spec.shape)))
        return tuple(out_set)

def generate_batch(dset, bs):
    while True:
        accum = [None, None]
        ib = 0
        while ib < bs:
            ib += 1
            x, y = dset[0]
            accum[0] = (x if accum[0] is None else numpy.vstack((accum[0], x)))
            accum[1] = (y if accum[1] is None else numpy.vstack((accum[1], y)))
            if ib >= bs:
                break
        yield (tf.convert_to_tensor(accum[0], dtype=numpy.float32), tf.convert_to_tensor(accum[1], dtype=numpy.float32))

def make_run():
    numpy.random.seed(0)

    all_idxs = numpy.arange(len(raw_data.root.meta))
    unique_peps = numpy.unique(list(map(lambda x: x.decode("utf-8"), raw_data.root.meta[all_idxs]['seq'])))
    n_data = unique_peps.shape[0]
    all_pep_idxs = numpy.arange(n_data)
    numpy.random.shuffle(all_pep_idxs)

    train_peptides = all_pep_idxs[:int(0.8*len(all_pep_idxs))]
    test_peptides = all_pep_idxs[int(0.8*len(all_pep_idxs)):int(0.9*len(all_pep_idxs))]
    valid_peptides = all_pep_idxs[int(0.9*len(all_pep_idxs)):]

    all_n = min(all_idxs.shape[0], MAX_SIZE)

    raw_data_seqs = list(map(lambda x: x.decode('utf-8'), raw_data.root.meta[:]['seq']))

    idxs = numpy.argwhere(numpy.isin(raw_data_seqs, unique_peps[train_peptides]))[:,0]
    numpy.random.shuffle(idxs)
    idxs = idxs[:int(all_n * 0.8)]
    test_idxs = numpy.argwhere(numpy.isin(raw_data_seqs, unique_peps[test_peptides]))[:,0]
    numpy.random.shuffle(test_idxs)
    test_idxs = test_idxs[:int(all_n * 0.1)]
    valid_idxs = numpy.argwhere(numpy.isin(raw_data_seqs, unique_peps[valid_peptides]))[:,0]
    numpy.random.shuffle(valid_idxs)
    valid_idxs = valid_idxs[:int(all_n * 0.1)]

    pep_lengths_tr = list(map(lambda x: len(x.decode('utf-8').replace("_", "").replace("M(ox)", "1")), raw_data.root.meta[idxs]['seq']))
    pep_lengths_va = list(map(lambda x: len(x.decode('utf-8').replace("_", "").replace("M(ox)", "1")), raw_data.root.meta[valid_idxs]['seq']))
    pep_lengths_te = list(map(lambda x: len(x.decode('utf-8').replace("_", "").replace("M(ox)", "1")), raw_data.root.meta[test_idxs]['seq']))

    dataset = MsDataset(raw_data, idxs, epoch_len=None)
    dataset_test = MsDataset(raw_data, test_idxs, epoch_len=None)

    steps = len(dataset) // BATCH_SIZE
    test_steps = len(dataset_test) // BATCH_SIZE

    last_epoch = 9999999

    pm2.fit_generator(generator=generate_batch(dataset, BATCH_SIZE), validation_data=generate_batch(dataset_test, BATCH_SIZE), steps_per_epoch=steps, validation_steps=test_steps, epochs=N_EPS if REPROD else last_epoch, verbose=1, use_multiprocessing=False, workers=0) # NOTE: Do NOT use multiprocessing when accessing hdf5 files because pytables breaks hard in this scenario.
    if REPROD:
        pm2.save(SAVE)

if __name__ == '__main__':
    make_run()
