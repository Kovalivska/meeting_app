# Meeting Transcriber & Translator

A Streamlit application for transcribing meeting recordings, translating text, and generating summaries using Whisper, free translation models, and OpenAI.

## Features

- **Audio Upload**: Support for multiple formats (MP3, WAV, M4A, MP4, FLAC, OGG)
- **Speech-to-Text**: Transcription using OpenAI Whisper
- **Translation**: 
  - **Free Translation**: Using Hugging Face Transformers (completely free)
  - **Google Translate**: Using googletrans library
- **Summary Generation**: 
  - **Simple Summary**: Basic text extraction (free)
  - **AI Summary**: Advanced analysis using OpenAI GPT (requires API key)
- **Download Results**: Export transcriptions, translations, and summaries

## Installation

1. **Install FFmpeg** (required for audio processing):
   ```bash
   # On macOS:
   brew install ffmpeg
   
   # On Ubuntu/Debian:
   sudo apt update && sudo apt install ffmpeg
   
   # On Windows:
   # Download from https://ffmpeg.org/download.html
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run meeting_transcriber_app.py
   ```

## Usage

1. **Upload Audio**: Select your meeting recording file (up to 3 hours)
2. **Choose Languages**: Set source and target languages (Ukrainian now supported!)
3. **Select Translation Service**: Choose between Free (Transformers) or Google Translate
4. **Transcribe**: Click "Transcribe Audio" to convert speech to text
5. **Translate**: (Optional) Translate text using your chosen service
6. **Summarize**: Use Simple Summary (free) or AI Summary (requires OpenAI API key)
7. **Download**: Save results using download buttons in each tab

## Translation Options

- **Free (Transformers)**: Uses Facebook's NLLB model via Hugging Face Transformers
  - Completely free
  - Works offline after initial model download
  - Supports 13+ languages including Ukrainian
  
- **Google Translate**: Uses googletrans library
  - May have better translation quality
  - Requires internet connection
  - May have usage limits

## Summary Options

- **Simple Summary**: 
  - Basic text extraction and statistics
  - Completely free, no API keys required
  - Shows key points and content metrics
  
- **AI Summary**: 
  - Advanced meeting analysis using OpenAI GPT
  - Structured summary with topics, decisions, action items
  - Requires OpenAI API key

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- Internet connection (for model downloads and some translation services)
- OpenAI API key (optional, only for AI summary generation)

## Supported Languages

**All services**: English, Russian, Ukrainian, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi

**Note**: Ukrainian language support is now available for both transcription and translation!

## Troubleshooting

### FFmpeg Error
If you see "No such file or directory: 'ffmpeg'":
1. Install FFmpeg using the commands above
2. Restart your terminal/command prompt
3. Verify installation: `ffmpeg -version`

### Translation Model Download
First time using Free Translation may take time to download the model (~2.5GB). Subsequent uses will be faster.

## Technical Details

- **Whisper model**: "small" (good balance of speed/accuracy)
- **Translation model**: facebook/nllb-200-distilled-600M
- **Audio processing**: Automatic conversion to WAV format for best compatibility
- **Text chunking**: Large texts are split for optimal processing