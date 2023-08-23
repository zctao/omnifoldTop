import os

import histogramming as myhu

import logging
logger = logging.getLogger("collectHistograms")

def collect_histograms(
    histograms_dir, # str, top directories to collect histograms
    observables=[], # list of str, names of observables to compute corrections
    histogram_suffix = "_histograms.root",
    output_name = None
    ):

    fpaths_histogram = []
    histograms_d = {}

    for hist_dir in histograms_dir:
        if not os.path.isdir(hist_dir):
            logger.error(f"Cannot access directory {hist_dir}")
            continue

        logger.info(f"Collect histogram files from {hist_dir}")

        for r, d, files in os.walk(hist_dir):
            for fname in files:
                if fname.endswith(histogram_suffix):
                    logger.debug(f" {os.path.join(r,fname)}")
                    fpaths_histogram.append(os.path.join(r,fname))

    if not fpaths_histogram:
        logger.error(f"Found no histogram file in {histograms_dir}")
        return histograms_d

    logger.info("Read histograms from files")
    for fpath in fpaths_histogram:
        hists_file_d = myhu.read_histograms_dict_from_file(fpath)

        if not observables:
            observables = list(hists_file_d.keys())

        for ob in observables:

            if not ob in hists_file_d:
                logger.error(f"Cannot find histograms for {ob} in {fpath}")
                continue

            if not ob in histograms_d:
                histograms_d[ob] = {}

            for hname in hists_file_d[ob]:
                if hname.startswith("Acceptance_") or hname.startswith("Efficiency_"):
                    continue

                if not hname in histograms_d[ob]:
                    histograms_d[ob][hname] = hists_file_d[ob][hname]
                else:
                    histograms_d[ob][hname] += hists_file_d[ob][hname]

    if output_name:
        logger.info(f"Write histograms to file {output_name}")

        outdir = os.path.dirname(output_name)
        if not os.path.isdir(outdir):
            logger.info(f"Make output directory {outdir}")
            os.makedirs(outdir)

        myhu.write_histograms_dict_to_file(histograms_d, output_name)

    return histograms_d

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('histograms_topdir', nargs='+', type=str,
                        help="Top directories to collect histograms")
    parser.add_argument('--observables', nargs='+', type=str,
                        help="List of observables. Use all that are available in the histograms if not specified")
    parser.add_argument('-s', '--suffix', type=str, default="_histograms.root",
                        help="Suffix to match histogram file names")
    parser.add_argument('-o', '--output-name', type=str, default="histograms_merged.root",
                        help="Name of the output file")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set the logging level to DEBUG.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    collect_histograms(
        args.histograms_topdir,
        observables = args.observables,
        histogram_suffix = args.suffix,
        output_name=args.output_name
    )