# Vocatio Lite

A streamlined audio transcription application with speaker diarization capabilities.

![Vocatio Lite](https://i.imgur.com/1AZ7KVf.png)

## Features

- **Transcription**: Convert audio/video files to text using OpenAI's Whisper API
- **Speaker Diarization**: Identify and label different speakers in the audio
- **Multilingual Support**: Transcribe in English and Swedish
- **Export Options**: Download transcripts as DOCX files
- **Speaker Statistics**: View speaking time for each participant

## Requirements

- Python 3.8 or higher
- OpenAI API key
- HuggingFace token (for speaker diarization)
- ffmpeg (optional, for handling large files)

## Installation

1. Clone the repository or download the source code
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Setting Up API Keys

### OpenAI API Key

1. Create an account at [OpenAI](https://platform.openai.com/)
2. Generate an API key in your account settings
3. Set the key as an environment variable:
   ```
   export OPENAI_API_KEY=your_key_here  # On Windows: set OPENAI_API_KEY=your_key_here
   ```

### HuggingFace Token (for Speaker Diarization)

1. Create an account at [HuggingFace](https://huggingface.co/)
2. Generate a token in your account settings
3. Accept the user conditions for both models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Set the token as an environment variable or enter it in the app interface:
   ```
   export HF_AUTH_TOKEN=your_token_here  # On Windows: set HF_AUTH_TOKEN=your_token_here
   ```

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```

The application will open in your default web browser.

## Deploying to Streamlit Cloud

To deploy this application to Streamlit Cloud:

1. Create an account at [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Set up your API keys in Streamlit Cloud secrets:
   - Go to the app settings ⚙️ > Secrets
   - Add the following keys:
     ```
     OPENAI_API_KEY = "your_openai_key_here"
     HF_AUTH_TOKEN = "your_huggingface_token_here"
     APP_PASSWORD = "your_chosen_password_here"
     ```
4. Make sure to add `ffmpeg` in your environment by adding a `packages.txt` file with:
   ```
   ffmpeg
   ```

## How to Use

1. Select your language preference (English or Swedish)
2. Toggle speaker diarization on or off
3. Upload an audio or video file
4. Click "Transcribe" to process the file
5. View the transcription results with speaker labels
6. Download the transcript as a DOCX file or copy to clipboard

## Limitations

- Maximum file size: 25MB (OpenAI API limit)
- Larger files are automatically compressed
- Speaker diarization accuracy varies based on audio quality
- Requires internet connection for API access

## Acknowledgements

- Developed based on the original Vocatio application
- Uses [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) for transcription
- Speaker diarization powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Built with [Streamlit](https://streamlit.io/) for the user interface 