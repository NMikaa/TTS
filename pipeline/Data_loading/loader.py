import os
import logging
import wandb
import pandas as pd


class DataLoader:
    def __init__(
            self,
            input_dir,
            output_dir,
            wandb_project="TTS_opensource",
            wandb_entity=None,
    ):
        """
        Parameters
        ----------
        input_dir : str
            Directory containing Common Voice TSV files 
            (train.tsv, test.tsv, dev.tsv, validated.tsv, other.tsv, clip_durations.tsv).
        output_dir : str
            Directory to save the final combined TSV file.
        wandb_project : str
            W&B project name for logging artifacts.
        wandb_entity : str
            W&B entity (user or team).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

    def run(self):
        """
        Main pipeline method for Data Loading:
          - Initialize W&B run
          - Load and merge all necessary TSV files into one big DataFrame
          - Optionally merge with clip_durations.tsv
          - Drop duplicates
          - Save single 'complete_data.tsv'
          - Log artifact to W&B (metadata: row count, file size, final path)
        """
        # -----------------------------
        # 1) Initialize a wandb run
        # -----------------------------
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name="data_loading",
            job_type="data_loading"
        )

        logging.info("Data loading pipeline started.")

        # -----------------------------
        # 2) Gather TSV files
        # -----------------------------
        files_to_include = ["train.tsv", "test.tsv", "dev.tsv", "validated.tsv", "other.tsv"]
        dfs = []

        for filename in files_to_include:
            path = os.path.join(self.input_dir, filename)
            if os.path.isfile(path):
                logging.info(f"Reading {filename}")
                df_part = pd.read_csv(path, sep="\t")
                df_part["source_file"] = filename  # track origin
                dfs.append(df_part)
            else:
                logging.warning(f"{filename} not found in {self.input_dir}, skipping...")

        if not dfs:
            logging.error("No valid TSV files found to merge. Exiting.")
            run.finish()
            return

        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Initial combined shape (no durations merged): {combined_df.shape}")

        # -----------------------------
        # 3) Merge with clip_durations (if exists)
        # -----------------------------
        clip_durations_path = os.path.join(self.input_dir, "clip_durations.tsv")
        if os.path.isfile(clip_durations_path):
            logging.info("Merging clip_durations.tsv")
            durations_df = pd.read_csv(clip_durations_path, sep="\t")
            # Standardize column name
            durations_df.columns = ["path", "duration_ms"]
            combined_df = pd.merge(combined_df, durations_df, on="path", how="left")
        else:
            logging.warning("clip_durations.tsv not found. Proceeding without durations.")

        # -----------------------------
        # 4) Drop duplicates by 'path'
        # -----------------------------
        before_drop = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset="path")
        after_drop = len(combined_df)
        logging.info(
            f"Dropped {before_drop - after_drop} duplicates based on 'path'. "
            f"Final shape: {combined_df.shape}"
        )

        # -----------------------------
        # 5) Save as complete_data.tsv
        # -----------------------------
        os.makedirs(self.output_dir, exist_ok=True)
        out_filename = "complete_data.tsv"
        out_path = os.path.join(self.output_dir, out_filename)

        combined_df.to_csv(out_path, sep="\t", index=False)
        logging.info(f"Saved final dataset to: {out_path}")

        # -----------------------------
        # 6) Log artifact to W&B
        # -----------------------------
        artifact = wandb.Artifact(
            name="data_loading",
            type="dataset",
            description="Complete merged dataset from Common Voice (no cleaning)."
        )

        artifact.add_file(out_path)

        # Attach metadata
        file_size = os.path.getsize(out_path)  # in bytes
        num_rows = len(combined_df)
        artifact.metadata = {
            "file_path": out_path,
            "size_bytes": file_size,
            "num_rows": num_rows
        }

        run.log_artifact(artifact)
        logging.info(
            f"W&B artifact logged: data_loading (Rows: {num_rows}, Size: {file_size} bytes)"
        )

        # -----------------------------
        # 7) Finish run
        # -----------------------------
        run.finish()
