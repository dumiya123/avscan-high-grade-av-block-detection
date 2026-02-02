"""
AV Block Detection System - CLI Orchestrator

This script provides a command-line interface for the entire pipeline:
- download: Download LUDB and PTB-XL datasets
- preprocess: Process raw ECG data and create 5-class labels
- train: Train the segmentation + classification model
- evaluate: Compute metrics on test set
- inference: Run prediction on new ECG with XAI
- report: Generate clinical PDF report
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def download_command(args):
    """Download datasets from PhysioNet"""
    from src.data.download import download_ludb, download_ptbxl
    
    print("=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)
    
    if 'ludb' in args.datasets:
        print("\n[1/2] Downloading LUDB...")
        download_ludb(output_dir=args.output_dir)
    
    if 'ptbxl' in args.datasets:
        print("\n[2/2] Downloading PTB-XL...")
        download_ptbxl(output_dir=args.output_dir)
    
    print("\nDownload complete!")


def preprocess_command(args):
    """Preprocess ECG data and create labels"""
    from src.data.preprocessing import preprocess_datasets
    
    print("=" * 60)
    print("PREPROCESSING ECG DATA")
    print("=" * 60)
    
    preprocess_datasets(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        validate=args.validate
    )
    
    print("\nPreprocessing complete!")


def train_command(args):
    """Train the model"""
    from src.training.train import train_model
    
    print("=" * 60)
    print("TRAINING AV BLOCK DETECTION MODEL")
    print("=" * 60)
    
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
    
    print("\nTraining complete!")


def evaluate_command(args):
    """Evaluate model on test set"""
    from src.training.evaluate import evaluate_model
    
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    evaluate_model(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    print("\nEvaluation complete!")


def inference_command(args):
    """Run inference on new ECG"""
    from src.inference.predictor import AVBlockPredictor
    import numpy as np
    
    print("=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    
    # Load predictor
    predictor = AVBlockPredictor(checkpoint=args.checkpoint)
    
    # Load ECG signal
    if args.input.endswith('.npy'):
        ecg_signal = np.load(args.input)
    else:
        raise ValueError("Input must be .npy file")
    
    # Run prediction
    result = predictor.predict(ecg_signal, generate_report=True)
    
    # Print summary
    print(f"\nRESULTS:")
    print(f"  Diagnosis: {result['diagnosis']['av_block_type']}")
    print(f"  Confidence: {result['diagnosis']['confidence']:.2%}")
    print(f"  P:QRS Ratio: {result['intervals']['p_qrs_ratio']:.2f}")
    print(f"\nExplanation:\n  {result['xai']['explanation']}")
    
    # Save report
    if args.output:
        predictor.save_report(result, args.output)
        print(f"\nReport saved to: {args.output}")


def report_command(args):
    """Generate clinical report from saved results"""
    from src.reports.report_generator import ClinicalReport
    import json
    
    print("=" * 60)
    print("GENERATING CLINICAL REPORT")
    print("=" * 60)
    
    # Load results
    with open(args.input, 'r') as f:
        result = json.load(f)
    
    # Generate report
    report = ClinicalReport()
    report.create_report(result, output_path=args.output)
    
    print(f"\nReport saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="AV Block Detection System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download datasets
  python run.py download --datasets ludb ptbxl
  
  # Preprocess data
  python run.py preprocess --validate
  
  # Train model
  python run.py train --epochs 50 --batch-size 16
  
  # Evaluate
  python run.py evaluate --checkpoint checkpoints/best_model.pth
  
  # Run inference
  python run.py inference --checkpoint checkpoints/best_model.pth --input sample.npy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['ludb', 'ptbxl'],
        default=['ludb', 'ptbxl'],
        help='Datasets to download'
    )
    download_parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for raw data'
    )
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess ECG data')
    preprocess_parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Directory with raw data'
    )
    preprocess_parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    preprocess_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate preprocessing results'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data'
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--seg-weight',
        type=float,
        default=0.6,
        help='Weight for segmentation loss'
    )
    train_parser.add_argument(
        '--clf-weight',
        type=float,
        default=0.4,
        help='Weight for classification loss'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    train_parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=15,
        help='Early stopping patience'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    eval_parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data'
    )
    eval_parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    inference_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input ECG file (.npy)'
    )
    inference_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output report path (.pdf)'
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate clinical report')
    report_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input results file (.json)'
    )
    report_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output report path (.pdf)'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    command_map = {
        'download': download_command,
        'preprocess': preprocess_command,
        'train': train_command,
        'evaluate': evaluate_command,
        'inference': inference_command,
        'report': report_command
    }
    
    try:
        command_map[args.command](args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
