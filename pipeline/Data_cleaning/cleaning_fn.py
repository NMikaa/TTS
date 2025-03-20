import os
import librosa
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
from pyannote.audio import Model


def trim_with_vad(
    input_dir: str,
    output_dir: str,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
    file_ext: str = ".wav"
):
    """
    Recursively finds .wav files in 'input_dir', trims leading/trailing silence
    by detecting voice activity using pyannote, and saves the trimmed files
    to 'output_dir' with the same folder structure.

    Args:
        input_dir (str): Path to the input folder containing .wav files.
        output_dir (str): Path to the output folder where trimmed files are saved.
        model_checkpoint (str): Pretrained model checkpoint for segmentation or VAD.
                                Default is "pyannote/voice-activity-detection".
        min_duration_on (float): Remove speech segments shorter than this many seconds
                                 (applied by pyannote pipeline).
        min_duration_off (float): Fill non-speech gaps shorter than this many seconds
                                  (applied by pyannote pipeline).
        file_ext (str): File extension to look for (default ".wav").

    Example usage:
        trim_with_vad(
            input_dir="/path/to/raw_wavs",
            output_dir="/path/to/trimmed_wavs",
            model_checkpoint="pyannote/voice-activity-detection",
            min_duration_on=0.0,
            min_duration_off=0.0,
            file_ext=".wav"
        )
    """
    model = Model.from_pretrained(
  "pyannote/segmentation-3.0")

    # 1) Initialize the VAD pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off
    })

    # 2) Walk through all files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in tqdm(files):
            if filename.lower().endswith(file_ext.lower()):
                input_path = os.path.join(root, filename)

                # Build the output path (keep same subfolders + filename)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # 3) Load audio as float and convert to int16 for pydub
                    audio_data, sr = librosa.load(input_path, sr=None, mono=True)
                    audio_data_int16 = np.int16(audio_data * 32767)

                    # Create a single PyDub AudioSegment
                    full_audio_segment = AudioSegment(
                        data=audio_data_int16.tobytes(),
                        sample_width=2,   # 16 bits
                        frame_rate=sr,
                        channels=1
                    )

                    # 4) Run VAD to get speech segments
                    vad_result = pipeline(input_path)
                    speech_timeline = vad_result.get_timeline()

                    if len(speech_timeline) == 0:
                        # No speech found at all
                        print(f"No speech found in {input_path}. Skipping.")
                        continue

                    # 5) Find earliest speech start and latest speech end
                    #    speech_timeline is sorted by time, so:
                    earliest_start = speech_timeline[0].start
                    latest_end = speech_timeline[-1].end

                    # 6) Trim the audio by slicing out the speech region
                    start_ms = earliest_start * 1000
                    end_ms = latest_end * 1000
                    trimmed_segment = full_audio_segment[start_ms:end_ms]

                    # 7) Save the trimmed audio to output_path
                    trimmed_segment.export(output_path, format="wav")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")