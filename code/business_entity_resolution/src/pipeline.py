import sys, os
import polars as pl
from multilingual_normalization import normalize_multilingual_name, extract_international_postal
from blocking import MultiViewMultilingualBlocker
from cascade_ranker import compute_structured_features
from tripartite_graph_consensus import TripartiteGraphConsensus

def run_pipeline(dataset_dir: str, output_dir: str):
    print("[INFO] Executing Pipeline...")
    os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "dataset", sys.argv[2] if len(sys.argv) > 2 else "output")