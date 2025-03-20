import os
import argparse
from dotenv import load_dotenv
from .loader import DataLoader
def main():
    # 1. Load environment variables from .env (which should hold WANDB_API_KEY)
    load_dotenv()  # Loads environment variables from .env into os.environ
    
    if "WANDB_API_KEY" not in os.environ:
        print("Warning: WANDB_API_KEY not found in environment. Make sure .env is set.")
    
    # 2. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Runner for Data Loading")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"C:\Users\LEGION\TTS\cv-corpus-21.0-2025-03-14\ka",
        help="Directory containing Common Voice TSV files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\LEGION\TTS\output",
        help="Directory to save the final combined TSV file."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="TTS_opensource",
        help="Weights & Biases project name."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (user or team)."
    )
    
    args = parser.parse_args()
    
    # 3. Instantiate the DataLoader class and run
    loader = DataLoader(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    loader.run()

if __name__ == "__main__":
    main()
