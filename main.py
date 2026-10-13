"""
Main script for the Spotify Skip Prediction project.
Runs the research pipeline: Ingest → EDA → Feature Engineering → Experiments → Interpretation.
"""
import time

from src.ingest import run_ingest_pipeline
from src.eda import run_eda_pipeline
from src.features import run_feature_engineering
from src.experiments import run_all_experiments
from src.interpretation import run_interpretation_pipeline


def main():
    start = time.time()
    print("=" * 80)
    print("SPOTIFY SKIP PREDICTION & AUDIO BEHAVIOR ANALYSIS")
    print("=" * 80)

    steps = [
        ("Data Ingestion & Matching", run_ingest_pipeline),
        ("Exploratory Data Analysis", run_eda_pipeline),
        ("Feature Engineering", run_feature_engineering),
        ("Model Training & Experiments", run_all_experiments),
        ("Model Interpretation & Case Studies", run_interpretation_pipeline),
    ]

    for name, func in steps:
        t0 = time.time()
        print(f"\n>>> STARTING: {name}")
        func()
        print(f">>> COMPLETED: {name} in {time.time() - t0:.2f}s\n")

    print("=" * 80)
    print(f"PIPELINE COMPLETE IN {time.time() - start:.2f}s")
    print("Figures: results/figures/  |  Tables: results/tables/")
    print("=" * 80)


if __name__ == "__main__":
    main()
