import re
import numpy as np
import radbin as r
from optparse import OptionParser
from xformats.detformats import read_mask
from xformats.matformats import write_mat
from xformats.yamlformats import read_ydat

description="Calculate radial indices from the center and mask."

usage="%prog -c xxx.xx,yyy.yy -m mask.png [-o <outputfile.mat>]"

defout='radind.mat'

def parse_center(center_str):
    mob = re.match(' *([0-9.]+)[,]([0-9.]+) *', center_str)
    if mob is None or len(mob.groups()) != 2:
        return None
    else:
        return (float(mob.group(1)), float(mob.group(2)))


def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-m", "--maskfile",
        action="store", type="string", dest="maskfile", default=None)
    oprs.add_option("-c", "--center",
        action="store", type="string", dest="cen_str", default=None)
    oprs.add_option("-o", "--output",
        action="store", type="string", dest="outfile", default=defout,
        help="Output file containing the indices. Default is '%s'" % defout)
    oprs.add_option("-f", "--force",
        action="store_true", dest="overwrite", default=False,
        help="Overwrite existing output file.")
    (opts, args) = oprs.parse_args()

    cen = None
    if opts.cen_str is not None:
        cen = parse_center(opts.cen_str)
        if cen is None:
            oprs.error("Could not parse center")
        print("Using " + str(cen) + " as center.")

    mask = None
    if opts.maskfile != None:
        mask = read_mask(opts.maskfile)
        print("Using '%s' as maskfile." % opts.maskfile)
    else:
        oprs.error("Maskfile must be given.")

    print("Will write indices to '%s'." % opts.outfile)
    radind = r.make_radind(mask.shape, cen, mask=mask)

#    write_pickle(radind, opts.outfile)
    write_mat('radind', opts.outfile, radind, overwrite=opts.overwrite)


if __name__ == "__main__":
    main()
