
"""
AV Block Detection System - Main CLI
Usage: python scripts/manage.py [command] [options]
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path to allow imports from src
# If running from ml_component root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports
from scripts.download_data import download_ludb, download_ptbxl

# We'll import other modules inside functions to avoid circular deps or early loading
# But basic logging setup can happen here
try:
    from src.utils.logger import setup_logging, get_logger
    logger = get_logger("manage")
except ImportError:
    # Fallback if utils not found or not setup
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("manage")

def download_command(args):
    """Download datasets."""
    logger.info("Starting dataset download...")
    
    if 'ludb' in args.datasets:
        download_ludb(output_dir=args.output_dir)
    if 'ptbxl' in args.datasets:
        download_ptbxl(output_dir=args.output_dir)

def preprocess_command(args):
    """Preprocess data."""
    # Correct import path
    from src.data_pipeline.preprocessing import preprocess_datasets
    
    logger.info("Starting preprocessing...")
    preprocess_datasets(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        validate=args.validate
    )

def train_command(args):
    """Train model."""
    # Correct import path
    from src.engine.trainer import train_model
    
    logger.info("Starting training...")
    train_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seg_weight=args.seg_weight,
        clf_weight=args.clf_weight,
        resume=args.resume,
        early_stop_patience=args.early_stop_patience
    )

def evaluate_command(args):
    """Evaluate model."""
    # Correct import path
    from src.engine.evaluator import evaluate_model
    
    logger.info("Starting evaluation...")
    evaluate_model(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

def inference_command(args):
    """Run inference."""
    # Correct import path
    from src.inference.predictor import AVBlockPredictor
    import numpy as np
    
    logger.info(f"Running inference on {args.input}")
    
    predictor = AVBlockPredictor(checkpoint=args.checkpoint)
    
    # Load input
    if args.input.endswith('.npy'):
        signal = np.load(args.input)
    else:
        logger.error("Only .npy files supported for now.")
        return

    result = predictor.predict(signal, generate_report=True)
    
    print("\n" + "="*40)
    print(f"DIAGNOSIS: {result['diagnosis']['av_block_type']}")
    print(f"CONFIDENCE: {result['diagnosis']['confidence']:.2%}")
    print("="*40 + "\n")
    
    if args.output:
        predictor.save_report(result, args.output)
        logger.info(f"Report saved to {args.output}")

def report_command(args):
    """Generate report from result JSON."""
    # Correct import path
    from src.reporting.generator import ClinicalReport
    import json
    
    with open(args.input, 'r') as f:
        result = json.load(f)
        
    report = ClinicalReport()
    report.create_report(result, output_path=args.output)
    logger.info(f"PDF Report generated at {args.output}")

def main():
    parser = argparse.ArgumentParser(description="AtrionNet Management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download
    down_parser = subparsers.add_parser('download', help='Download Data')
    down_parser.add_argument('--datasets', nargs='+', default=['ludb', 'ptbxl'])
    down_parser.add_argument('--output-dir', default='data/raw')
    
    # Preprocess
    prep_parser = subparsers.add_parser('preprocess', help='Preprocess Data')
    prep_parser.add_argument('--raw-dir', default='data/raw')
    prep_parser.add_argument('--output-dir', default='data/processed')
    prep_parser.add_argument('--validate', action='store_true')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train Model')
    train_parser.add_argument('--data-dir', default='data/processed')
    train_parser.add_argument('--checkpoint-dir', default='checkpoints')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--seg-weight', type=float, default=0.6)
    train_parser.add_argument('--clf-weight', type=float, default=0.4)
    train_parser.add_argument('--resume', default=None)
    train_parser.add_argument('--early-stop-patience', type=int, default=15)

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate Model')
    eval_parser.add_argument('--checkpoint', required=True)
    eval_parser.add_argument('--data-dir', default='data/processed')
    eval_parser.add_argument('--output-dir', default='results')

    # Inference
    inf_parser = subparsers.add_parser('inference', help='Run Inference')
    inf_parser.add_argument('--checkpoint', required=True)
    inf_parser.add_argument('--input', required=True)
    inf_parser.add_argument('--output', default=None)
    
    # Report
    rep_parser = subparsers.add_parser('report', help='Generate PDF Report')
    rep_parser.add_argument('--input', required=True)
    rep_parser.add_argument('--output', required=True)

    args = parser.parse_args()
    
    # Initialize logging if possible
    # setup_logging()
    
    if args.command:
        # Check command
        if args.command == 'download':
            download_command(args)
        elif args.command == 'preprocess':
            preprocess_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'inference':
            inference_command(args)
        elif args.command == 'report':
            report_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
