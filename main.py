"""
Main Orchestrator for the Spotify Skip Prediction Project.
Runs the complete research pipeline end-to-end or individual steps via CLI flags.
"""
import argparse
import sys
import time

from src.ingest import run_ingest_pipeline
from src.eda import run_eda_pipeline
from src.features import run_feature_engineering
from src.experiments import run_all_experiments
from src.interpretation import run_interpretation_pipeline
from src.cluster import run_clustering_pipeline
from src.recommend import run_recommender_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Spotify Skip Prediction Pipeline Orchestrator")
    parser.add_argument(
        "--step",
        choices=["all", "ingest", "eda", "features", "experiments", "interpret", "cluster", "recommend"],
        default="all",
        help="Pipeline step to execute (default: all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_total = time.time()
    print("=" * 80)
    print("SPOTIFY EXTENDED STREAMING HISTORY: SKIP PREDICTION & AUDIO BEHAVIOR ANALYSIS")
    print(f"Executing step: {args.step.upper()}")
    print("=" * 80)

    steps_to_run = []
    if args.step == "all":
        steps_to_run = [
            ("Ingest & Match Analysis", run_ingest_pipeline),
            ("Exploratory Data Analysis", run_eda_pipeline),
            ("Causal Feature Engineering", run_feature_engineering),
            ("Model Ladder & Experiments", run_all_experiments),
            ("Model Interpretation & Case Studies", run_interpretation_pipeline),
            ("Add-On 1: Listening Modes Clustering", run_clustering_pipeline),
            ("Add-On 2: Offline Recommender Simulation", run_recommender_pipeline),
        ]
    elif args.step == "ingest":
        steps_to_run = [("Ingest & Match Analysis", run_ingest_pipeline)]
    elif args.step == "eda":
        steps_to_run = [("Exploratory Data Analysis", run_eda_pipeline)]
    elif args.step == "features":
        steps_to_run = [("Causal Feature Engineering", run_feature_engineering)]
    elif args.step == "experiments":
        steps_to_run = [("Model Ladder & Experiments", run_all_experiments)]
    elif args.step == "interpret":
        steps_to_run = [("Model Interpretation & Case Studies", run_interpretation_pipeline)]
    elif args.step == "cluster":
        steps_to_run = [("Add-On 1: Listening Modes Clustering", run_clustering_pipeline)]
    elif args.step == "recommend":
        steps_to_run = [("Add-On 2: Offline Recommender Simulation", run_recommender_pipeline)]

    for name, func in steps_to_run:
        step_start = time.time()
        print(f"\n>>> STARTING: {name}")
        func()
        print(f">>> COMPLETED: {name} in {time.time() - step_start:.2f} seconds\n")

    print("=" * 80)
    print(f"PIPELINE EXECUTION COMPLETE IN {time.time() - start_total:.2f} SECONDS")
    print("Figures saved to: results/figures/")
    print("Tables saved to:  results/tables/")
    print("Cleaned data in:  data/")
    print("=" * 80)


if __name__ == "__main__":
    main()
