# src/data/fetch_seismogram.py
"""
CLI to run batch extractor (wraps src.features.extract_p_wave_feature.batch_extract)
"""
import argparse
from src.features.extract_p_wave_feature import batch_extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="Data/EEW_features_2024-10-21.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--net", type=str, default="IU")
    parser.add_argument("--stations", nargs="*", default=None)
    args = parser.parse_args()

    df = batch_extract(out_csv=args.out, num_samples=args.n, stations=args.stations, net=args.net)
    print("Saved:", args.out)
