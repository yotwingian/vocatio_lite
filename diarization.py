import torch
import os
import tempfile
from pathlib import Path
import streamlit as st
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Union

# Default HF access token for pyannote models
DEFAULT_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)


class Diarization:
    """Speaker diarization using pyannote.audio."""
    
    def __init__(self, device=None):
        """Initialize the diarization model.
        
        Args:
            device: The torch device to use. If None, will use cuda if available else cpu.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.pipeline = None
        self.auth_token = DEFAULT_AUTH_TOKEN
    
    def initialize(self, auth_token=None):
        """Initialize the diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
            
            # Use provided token or default from environment variable
            token = auth_token or self.auth_token
            
            if not token:
                st.error("No HuggingFace token provided. Please set HF_AUTH_TOKEN environment variable.")
                return False
            
            # Initialize pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            
            # Move pipeline to device
            self.pipeline.to(self.device)
            return True
            
        except ImportError:
            st.error("pyannote.audio is not installed.")
            return False
        except Exception as e:
            st.error(f"Error initializing diarization pipeline: {str(e)}")
            return False
    
    def process_audio(self, audio_file_path: str, num_speakers: Optional[int] = None) -> Dict:
        """Process an audio file and perform speaker diarization.
        
        Args:
            audio_file_path: Path to the audio file.
            num_speakers: Optional number of speakers. If None, the model will determine automatically.
            
        Returns:
            Dictionary with diarization results.
        """
        if self.pipeline is None:
            success = self.initialize()
            if not success:
                return {"error": "Failed to initialize diarization pipeline"}
        
        try:
            # Perform diarization
            with st.spinner("Performing speaker diarization... This may take a moment."):
                start_time = time.time()
                
                # If num_speakers is provided, set it
                diarization = self.pipeline(audio_file_path, num_speakers=num_speakers)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Process diarization results
                diarization_result = self._format_diarization_result(diarization)
                diarization_result["processing_time"] = processing_time
                
                return diarization_result
                
        except Exception as e:
            return {"error": f"Error during diarization: {str(e)}"}
    
    def _format_diarization_result(self, diarization):
        """Format the diarization result for easier use.
        
        Args:
            diarization: Raw diarization result from pyannote.
            
        Returns:
            Formatted diarization result.
        """
        segments = []
        speaker_times = {}
        speaker_mapping = {}  # To map original speaker IDs to sequential numbers
        current_speaker_idx = 0
        
        # Process each segment
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            duration = end - start
            
            # Map speaker to sequential number if not already mapped
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"{current_speaker_idx:02d}"  # Format with leading zero
                current_speaker_idx += 1
            
            speaker_id = speaker_mapping[speaker]
            
            # Add to segments
            segments.append({
                "start": start,
                "end": end,
                "speaker": speaker_id
            })
            
            # Update speaker times
            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = 0
            speaker_times[speaker_id] += duration
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        return {
            "segments": segments,
            "speaker_times": speaker_times,
            "num_speakers": len(speaker_times),
            "speaker_mapping": speaker_mapping
        }
    
    def align_transcript_with_diarization(self, transcript: str, diarization_result: Dict) -> List[Dict]:
        """Align transcript with diarization results.
        
        Args:
            transcript: Plain text transcript.
            diarization_result: Diarization result from process_audio.
            
        Returns:
            List of segments with speaker labels and text.
        """
        if "error" in diarization_result:
            return [{"speaker": "unknown", "text": transcript}]
        
        # This is a simplified approach. For a complete solution, 
        # you would need to align transcript with audio timestamps,
        # which requires word-level timestamps from the ASR system.
        
        # Split transcript into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        # Get segments and sort by start time
        segments = diarization_result["segments"]
        
        # Simple allocation of sentences to speakers
        aligned_segments = []
        
        # If we have more sentences than segments, we'll need to combine some sentences
        if len(sentences) <= len(segments):
            for i, sentence in enumerate(sentences):
                if i < len(segments):
                    aligned_segments.append({
                        "start": segments[i]["start"],
                        "end": segments[i]["end"],
                        "speaker": segments[i]["speaker"],
                        "text": sentence
                    })
                else:
                    # Append any remaining sentences to the last segment
                    aligned_segments[-1]["text"] += " " + sentence
        else:
            # More segments than sentences, distribute sentences across segments
            sentence_idx = 0
            for segment in segments:
                if sentence_idx < len(sentences):
                    aligned_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "speaker": segment["speaker"],
                        "text": sentences[sentence_idx]
                    })
                    sentence_idx += 1
        
        return aligned_segments
    
    def format_transcript_with_speakers(self, aligned_segments: List[Dict]) -> str:
        """Format the transcript with speaker labels.
        
        Args:
            aligned_segments: List of segments with speaker labels and text.
            
        Returns:
            Formatted transcript string.
        """
        formatted_transcript = "Transcription Result:\n\n"
        current_speaker = None
        max_length = 71  # Maximum line length
        
        # Calculate the prefix length for consistent alignment
        prefix_length = len("SPEAKER_00: ")
        
        for segment in aligned_segments:
            speaker = segment["speaker"]
            text = segment["text"].strip()
            
            # Format speaker label
            speaker_id = f"SPEAKER_{speaker}"
            
            # Add speaker label if different from previous
            if speaker != current_speaker:
                # Create prefix with the speaker ID followed by ": "
                line_prefix = f"{speaker_id}: "
                # Pad with spaces if needed to align all text consistently
                line_prefix = line_prefix.ljust(prefix_length)
                current_speaker = speaker
            else:
                # For continuation lines, just use spaces to align with text after the colon
                line_prefix = " " * prefix_length
            
            # Handle line wrapping for long text
            remaining_text = text
            
            while remaining_text:
                # Find where to split the line
                if len(remaining_text) <= max_length:
                    line = remaining_text
                    remaining_text = ""
                else:
                    # Find last space within max_length
                    split_at = remaining_text[:max_length].rfind(' ')
                    if split_at == -1:  # No space found, force split
                        split_at = max_length
                    
                    line = remaining_text[:split_at].strip()
                    remaining_text = remaining_text[split_at:].strip()
                
                # Add the line with appropriate prefix
                formatted_transcript += line_prefix + line + "\n"
                
                # Subsequent lines use the continuation prefix
                line_prefix = " " * prefix_length
        
        return formatted_transcript 