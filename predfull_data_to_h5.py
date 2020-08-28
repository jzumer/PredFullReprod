import numpy
import tables
import tqdm
import sys

if len(sys.argv) <= 2:
    print("USAGE: {} infile outfile".format(sys.argv[0]))
    sys.exit(-1)

PROT_BLIT_LEN = 2000*10
PROT_STR_LEN = 30

def blit(mz_list, itensity_list, mass, bin_size, charge): #spectra2vector
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

class Meta(tables.IsDescription):
    charge = tables.UInt8Col()
    mass = tables.Float32Col()
    lgt = tables.UInt8Col()
    seq = tables.StringCol(50)

out = tables.open_file(sys.argv[2], 'w')
meta_tbl = out.create_table(out.root, "meta", Meta, "Metadata")
data_tbl = out.create_earray(out.root, "spectrum", tables.Float32Atom(), (0, PROT_BLIT_LEN))

lines = open(sys.argv[1], "r").readlines()
fillme = None
skip = False
for l in tqdm.tqdm(lines, total=len(lines), desc="Processing"):
    l = l.strip()
    if l.startswith("BEGIN IONS"):
        fillme = meta_tbl.row
        spec_mz = []
        spec_intens = []
        skip = False
    else:
        if skip:
            continue
        else:
            if l.startswith("SCAN") or l.startswith("PROTEIN") or l.startswith("FILENAME") or l.startswith("COLLISION_ENERGY") or l.startswith("MSLEVEL") or l.startswith("SCORE") or l.startswith("FDR"):
                # NOTE: COLLISION_ENERGY is always 0 in the data at predfull.com, so we skip that line and set energy to 25 NCE in the dataset.
                continue
            elif l.startswith("PEPMASS"):
                fillme['mass'] = float(l.split("=")[1])
            elif l.startswith("SEQ"):
                seq = l.split("=")[1]
                lgt = len(seq)
                if lgt > PROT_STR_LEN:
                    skip = True
                    continue
                else:
                    fillme['lgt'] = lgt
                    fillme['seq'] = seq
            elif l.startswith("CHARGE"):
                charge = l.split("=")[1].replace("+", "")
                charge = int(charge)
                if (charge >= 2) and (charge <= 3):
                    fillme['charge'] = charge
                else:
                    skip = True
                    continue
            elif l.startswith("END IONS"):
                fillme.append()
                data_tbl.append(blit(spec_mz, spec_intens, fillme['mass'], 0.1, charge).reshape((1, -1)))
                fillme = None
            elif len(l) == 0:
                continue
            else:
                mz, intens = l.split("\t")
                mz = float(mz)
                intens = float(intens)
                spec_mz.append(mz)
                spec_intens.append(intens)
out.flush()
