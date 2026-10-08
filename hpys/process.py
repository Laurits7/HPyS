"""Run the HPS tau builder on a parquet ntuple and write the input fields plus the HPS output."""

import argparse
import os

import awkward as ak
from omegaconf import OmegaConf

from hpys.hps import HPSTauBuilder

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "hps.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input parquet file with reco jets and candidates")
    parser.add_argument("output_file", help="Output parquet file")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="HPS configuration (yaml)")
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    args = parser.parse_args()

    data = ak.from_parquet(args.input_file)
    builder = HPSTauBuilder(cfg=OmegaConf.load(args.config), verbosity=args.verbosity)
    processed_data = builder.process_jets(data)

    data_to_save = {field: data[field] for field in data.fields}
    data_to_save.update(processed_data)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    ak.to_parquet(ak.Array(data_to_save), args.output_file)


if __name__ == "__main__":
    main()
