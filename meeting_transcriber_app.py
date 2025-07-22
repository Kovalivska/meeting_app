import streamlit as st
import tempfile
import os
import whisper
import openai
from pydub import AudioSegment
import subprocess
import re
import requests

# Try to import googletrans, fallback if not available
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    st.warning("Google Translate library not available. Using fallback translator.")

# Set page configuration
st.set_page_config(
    page_title="Meeting Transcriber & Translator",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Meeting Transcriber & Translator")
st.markdown("Upload meeting recordings for transcription, translation, and summary generation")

# Initialize session state variables
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'translation' not in st.session_state:
    st.session_state.translation = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Language mappings
language_names = {
    "auto": "Auto-detect",
    "en": "English", 
    "ru": "Russian", 
    "uk": "Ukrainian",
    "es": "Spanish", 
    "fr": "French", 
    "de": "German", 
    "it": "Italian",
    "pt": "Portuguese", 
    "zh": "Chinese", 
    "ja": "Japanese",
    "ko": "Korean", 
    "ar": "Arabic", 
    "hi": "Hindi"
}

# Source language selection
source_language = st.sidebar.selectbox(
    "Meeting language:",
    list(language_names.keys()),
    index=0,
    format_func=lambda x: language_names[x],
    help="Select the language of the meeting recording"
)

# Target language for translation
target_language = st.sidebar.selectbox(
    "Translation language:",
    [k for k in language_names.keys() if k != "auto"],
    index=0,
    format_func=lambda x: language_names[x],
    help="Select the language for translation and summary"
)

# OpenAI API key input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (optional):",
    type="password",
    help="Enter your OpenAI API key for AI-powered summary generation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Supported formats:** MP3, WAV, M4A, MP4, FLAC, OGG")
st.sidebar.markdown("**Max duration:** 3 hours")

# Function to check ffmpeg
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ffmpeg_instructions():
    st.error("FFmpeg is not installed. Please install it:")
    st.code("""
    # On macOS:
    brew install ffmpeg
    
    # On Ubuntu/Debian:
    sudo apt update && sudo apt install ffmpeg
    
    # On Windows:
    # Download from https://ffmpeg.org/download.html
    """)

def translate_with_fallback(text, target_lang):
    """Fallback translation using basic word replacement"""
    # Basic word mappings for common languages
    basic_translations = {
        "uk": {
            "meeting": "зустріч", "question": "питання", "decision": "рішення",
            "important": "важливо", "discussion": "обговорення", "problem": "проблема",
            "solution": "рішення", "task": "завдання", "goal": "мета"
        },
        "ru": {
            "meeting": "встреча", "question": "вопрос", "decision": "решение",
            "important": "важно", "discussion": "обсуждение", "problem": "проблема",
            "solution": "решение", "task": "задача", "goal": "цель"
        }
    }
    
    if target_lang in basic_translations:
        translated_text = text
        for en_word, translated_word in basic_translations[target_lang].items():
            translated_text = translated_text.replace(en_word, translated_word)
        return translated_text
    
    return text  # Return original if no translation available

def translate_with_google(text, target_lang):
    """Fast and reliable Google Translate with fallback"""
    try:
        if GOOGLETRANS_AVAILABLE:
            translator = Translator()
            
            # For long texts, split into smaller chunks
            if len(text) > 3000:
                chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
                translated_chunks = []
                
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        translation = translator.translate(chunk, dest=target_lang)
                        translated_chunks.append(translation.text)
                        progress_bar.progress((i + 1) / len(chunks))
                progress_bar.empty()
                
                return " ".join(translated_chunks)
            else:
                # Short text - translate directly
                translation = translator.translate(text, dest=target_lang)
                return translation.text
        else:
            # Use fallback translation
            st.info("Using basic translation due to library limitations")
            return translate_with_fallback(text, target_lang)
            
    except Exception as e:
        st.warning(f"Google Translate unavailable, using fallback: {str(e)}")
        return translate_with_fallback(text, target_lang)

def generate_smart_summary(text, target_lang):
    """Generate intelligent structured summary with analysis"""
    try:
        # Clean and prepare text
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        if len(sentences) == 0:
            return "No meaningful content found for summarization."
        
        # Analyze text for different content types
        questions = []
        decisions = []
        key_points = []
        
        # Keywords for different categories
        question_keywords = ['что', 'как', 'когда', 'где', 'почему', 'кто', 'сколько', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'чи', 'як', 'коли', 'де', 'чому', 'хто', 'скільки']
        decision_keywords = ['решили', 'решение', 'принято', 'договорились', 'decided', 'decision', 'agreed', 'concluded', 'вирішили', 'рішення', 'прийнято', 'домовилися']
        important_keywords = ['важно', 'главное', 'основное', 'ключевое', 'проблема', 'задача', 'цель', 'important', 'main', 'key', 'problem', 'task', 'goal', 'важливо', 'головне', 'основне', 'ключове', 'проблема', 'завдання', 'мета']
        
        # Categorize sentences
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for questions
            if ('?' in sentence or 
                any(keyword in sentence_lower for keyword in question_keywords) or
                sentence.strip().endswith('?')):
                questions.append(sentence)
            
            # Check for decisions
            elif any(keyword in sentence_lower for keyword in decision_keywords):
                decisions.append(sentence)
            
            # Check for important points
            elif any(keyword in sentence_lower for keyword in important_keywords):
                key_points.append(sentence)
            
            # Add sentences with numbers, dates, or specific mentions
            elif (re.search(r'\d{1,2}[./]\d{1,2}', sentence) or  # dates
                  re.search(r'\d+%', sentence) or  # percentages
                  re.search(r'\$\d+|\d+\$', sentence) or  # money
                  len(sentence.split()) > 8):  # longer sentences are often important
                key_points.append(sentence)
        
        # If categories are empty, fill with key sentences
        if not questions and not decisions and not key_points:
            # Fallback: take sentences from different parts
            num_sentences = len(sentences)
            if num_sentences <= 6:
                key_points = sentences
            else:
                key_points = (sentences[:2] + 
                            sentences[num_sentences//3:num_sentences//3+2] + 
                            sentences[2*num_sentences//3:2*num_sentences//3+2] + 
                            sentences[-2:])
        
        # Limit number of items in each category
        questions = questions[:4]
        decisions = decisions[:4]
        key_points = key_points[:6]
        
        # Language-specific templates
        templates = {
            "en": {
                "title": "📋 Structured Meeting Summary",
                "questions_label": "❓ Questions Discussed",
                "key_points_label": "💡 Key Points & Topics",
                "decisions_label": "✅ Decisions & Agreements",
                "stats_label": "📊 Meeting Statistics"
            },
            "uk": {
                "title": "📋 Структуроване резюме зустрічі",
                "questions_label": "❓ Обговорені питання",
                "key_points_label": "💡 Ключові моменти та теми",
                "decisions_label": "✅ Рішення та домовленості",
                "stats_label": "📊 Статистика зустрічі"
            },
            "ru": {
                "title": "📋 Структурированное резюме встречи",
                "questions_label": "❓ Обсуждённые вопросы",
                "key_points_label": "💡 Ключевые моменты и темы",
                "decisions_label": "✅ Решения и договорённости",
                "stats_label": "📊 Статистика встречи"
            },
            "de": {
                "title": "📋 Strukturierte Meeting-Zusammenfassung",
                "questions_label": "❓ Diskutierte Fragen",
                "key_points_label": "💡 Wichtige Punkte und Themen",
                "decisions_label": "✅ Entscheidungen und Vereinbarungen",
                "stats_label": "📊 Meeting-Statistiken"
            },
            "es": {
                "title": "📋 Resumen Estructurado de Reunión",
                "questions_label": "❓ Preguntas Discutidas",
                "key_points_label": "💡 Puntos Clave y Temas",
                "decisions_label": "✅ Decisiones y Acuerdos",
                "stats_label": "📊 Estadísticas de la Reunión"
            },
            "fr": {
                "title": "📋 Résumé Structuré de Réunion",
                "questions_label": "❓ Questions Discutées",
                "key_points_label": "💡 Points Clés et Sujets",
                "decisions_label": "✅ Décisions et Accords",
                "stats_label": "📊 Statistiques de la Réunion"
            }
        }
        
        template = templates.get(target_lang, templates["en"])
        
        # Build the structured summary
        summary_parts = [f"**{template['title']}**\n"]
        
        # Add questions section
        if questions:
            summary_parts.append(f"## {template['questions_label']}")
            for i, question in enumerate(questions, 1):
                # Translate if needed
                if target_lang not in ["en"] and target_lang in templates:
                    try:
                        if GOOGLETRANS_AVAILABLE:
                            translator = Translator()
                            question = translator.translate(question, dest=target_lang).text
                        else:
                            question = translate_with_fallback(question, target_lang)
                    except:
                        question = translate_with_fallback(question, target_lang)
                summary_parts.append(f"{i}. {question}")
            summary_parts.append("")
        
        # Add key points section
        if key_points:
            summary_parts.append(f"## {template['key_points_label']}")
            for point in key_points:
                # Translate if needed
                if target_lang not in ["en"] and target_lang in templates:
                    try:
                        if GOOGLETRANS_AVAILABLE:
                            translator = Translator()
                            point = translator.translate(point, dest=target_lang).text
                        else:
                            point = translate_with_fallback(point, target_lang)
                    except:
                        point = translate_with_fallback(point, target_lang)
                summary_parts.append(f"• {point}")
            summary_parts.append("")
        
        # Add decisions section
        if decisions:
            summary_parts.append(f"## {template['decisions_label']}")
            for i, decision in enumerate(decisions, 1):
                # Translate if needed
                if target_lang not in ["en"] and target_lang in templates:
                    try:
                        if GOOGLETRANS_AVAILABLE:
                            translator = Translator()
                            decision = translator.translate(decision, dest=target_lang).text
                        else:
                            decision = translate_with_fallback(decision, target_lang)
                    except:
                        decision = translate_with_fallback(decision, target_lang)
                summary_parts.append(f"{i}. {decision}")
            summary_parts.append("")
        
        # Add statistics
        summary_parts.append(f"## {template['stats_label']}")
        summary_parts.append(f"• Общая длина текста: {len(text)} символов")
        summary_parts.append(f"• Количество предложений: {len(sentences)}")
        summary_parts.append(f"• Обнаружено вопросов: {len(questions)}")
        summary_parts.append(f"• Ключевых моментов: {len(key_points)}")
        summary_parts.append(f"• Принятых решений: {len(decisions)}")
        
        formatted_summary = "\n".join(summary_parts)
        
        # If no structured content was found, provide a basic summary
        if not questions and not decisions and not key_points:
            basic_points = sentences[:8]  # Take first 8 sentences
            summary_parts = [f"**{template['title']}**\n"]
            summary_parts.append(f"## {template['key_points_label']}")
            
            for point in basic_points:
                if target_lang not in ["en"] and target_lang in templates:
                    try:
                        if GOOGLETRANS_AVAILABLE:
                            translator = Translator()
                            point = translator.translate(point, dest=target_lang).text
                        else:
                            point = translate_with_fallback(point, target_lang)
                    except:
                        point = translate_with_fallback(point, target_lang)
                summary_parts.append(f"• {point}")
            
            formatted_summary = "\n".join(summary_parts)
        
        return formatted_summary
        
    except Exception as e:
        st.error(f"Summary generation error: {str(e)}")
        return f"Error generating summary: {str(e)}"

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file:",
        type=['mp3', 'wav', 'm4a', 'mp4', 'flac', 'ogg'],
        help="Upload your meeting recording (max 3 hours)"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "Size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "Type": uploaded_file.type
        }
        st.json(file_details)
        
        # Audio player
        st.audio(uploaded_file, format=uploaded_file.type)

with col2:
    st.header("🎯 Processing")
    
    if uploaded_file is not None:
        # Check ffmpeg before transcription
        if not check_ffmpeg():
            install_ffmpeg_instructions()
        else:
            # Transcription button
            if st.button("🎤 Transcribe Audio", type="primary", use_container_width=True):
                with st.spinner("Processing audio... This may take several minutes for long recordings."):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Convert audio if needed using pydub
                        try:
                            audio = AudioSegment.from_file(tmp_file_path)
                            # Convert to wav for better Whisper compatibility
                            wav_path = tmp_file_path.replace(tmp_file_path.split('.')[-1], 'wav')
                            audio.export(wav_path, format="wav")
                            os.unlink(tmp_file_path)  # Remove original
                            tmp_file_path = wav_path
                        except Exception as audio_error:
                            st.warning(f"Audio conversion warning: {str(audio_error)}")
                        
                        # Load Whisper model
                        model = whisper.load_model("small")
                        
                        # Transcribe audio
                        whisper_lang = None if source_language == "auto" else source_language
                        result = model.transcribe(tmp_file_path, language=whisper_lang)
                        
                        st.session_state.transcription = result["text"]
                        
                        # Store detected language for better translation
                        if source_language == "auto" and 'language' in result:
                            st.session_state.detected_language = result['language']
                        else:
                            st.session_state.detected_language = source_language
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                        st.success("✅ Transcription completed!")
                        
                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")
                        if "tmp_file_path" in locals():
                            try:
                                os.unlink(tmp_file_path)
                            except:
                                pass
        
        # Translation button
        if st.session_state.transcription and st.button("🌐 Translate Text", use_container_width=True):
            with st.spinner("Translating text..."):
                try:
                    translated_text = translate_with_google(
                        st.session_state.transcription, 
                        target_language
                    )
                    
                    st.session_state.translation = translated_text
                    st.success("✅ Translation completed!")
                    
                except Exception as e:
                    st.error(f"Error during translation: {str(e)}")
        
        # Summary generation buttons
        text_for_summary = st.session_state.translation if st.session_state.translation else st.session_state.transcription
        
        if text_for_summary:
            col_summary1, col_summary2 = st.columns(2)
            
            with col_summary1:
                if st.button("🧠 Smart Summary (Free)", use_container_width=True):
                    with st.spinner("Generating smart summary..."):
                        try:
                            smart_summary = generate_smart_summary(text_for_summary, target_language)
                            st.session_state.summary = smart_summary
                            st.success("✅ Smart summary generated!")
                        except Exception as e:
                            st.error(f"Error generating smart summary: {str(e)}")
            
            with col_summary2:
                if st.button("📋 AI Summary (OpenAI)", use_container_width=True):
                    if not openai_api_key:
                        st.error("Please enter your OpenAI API key in the sidebar for AI summary")
                    else:
                        with st.spinner("Generating AI summary..."):
                            try:
                                # Set OpenAI API key
                                openai.api_key = openai_api_key
                                
                                target_lang_name = language_names.get(target_language, target_language)
                                
                                prompt = f"""
                                Analyze the following meeting transcript and create a structured summary in {target_lang_name}.
                                
                                Please include these sections:
                                1. Main Topics Discussed
                                2. Key Decisions and Agreements
                                3. Action Items and Responsible Parties
                                4. Next Steps
                                
                                Meeting transcript:
                                {text_for_summary[:3000]}...
                                
                                Summary:
                                """
                                
                                # Generate summary using OpenAI API
                                response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=1500,
                                    temperature=0.7
                                )
                                
                                st.session_state.summary = response.choices[0].message.content
                                st.success("✅ AI Summary generated!")
                                
                            except Exception as e:
                                st.error(f"Error generating AI summary: {str(e)}")

# Results section
st.header("📄 Results")

tab1, tab2, tab3 = st.tabs(["Transcription", "Translation", "Summary"])

with tab1:
    if st.session_state.transcription:
        st.subheader("📝 Transcription")
        
        # Show detected language if available
        if hasattr(st.session_state, 'detected_language'):
            detected_lang_name = language_names.get(st.session_state.detected_language, st.session_state.detected_language)
            st.info(f"Detected language: {detected_lang_name}")
        
        st.text_area("Transcribed text:", st.session_state.transcription, height=300)
        
        # Download button
        st.download_button(
            "💾 Download Transcription",
            st.session_state.transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )
    else:
        st.info("Transcription will appear here after processing the audio file")

with tab2:
    if st.session_state.translation:
        st.subheader("🌐 Translation")
        st.text_area("Translated text:", st.session_state.translation, height=300)
        
        # Download button
        st.download_button(
            "💾 Download Translation",
            st.session_state.translation,
            file_name="translation.txt",
            mime="text/plain"
        )
    else:
        st.info("Translation will appear here after translating the transcription")

with tab3:
    if st.session_state.summary:
        st.subheader("📋 Meeting Summary")
        st.markdown(st.session_state.summary)
        
        # Download button
        st.download_button(
            "💾 Download Summary",
            st.session_state.summary,
            file_name="meeting_summary.txt",
            mime="text/plain"
        )
    else:
        st.info("Meeting summary will appear here after generation")

# Footer with author information
st.markdown("---")
st.markdown("""
🚀 **Meeting Analysis Tool** | Powered by Whisper, Google Translate & OpenAI

👩‍💻 **Created by:** [Svitlana Kovalivska](https://www.linkedin.com/in/svitlana-kovalivska) 

🙏 **Thank you for using this application!** Your feedback and suggestions are always welcome.

🔗 **Connect with me on LinkedIn:** [www.linkedin.com/in/svitlana-kovalivska](https://www.linkedin.com/in/svitlana-kovalivska)
""")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Install FFmpeg**: Make sure FFmpeg is installed on your system
    2. **Upload Audio**: Choose your meeting recording file (MP3, WAV, etc.)
    3. **Select Languages**: Choose source and target languages (Ukrainian supported!)
    4. **Transcribe**: Click "Transcribe Audio" to convert speech to text
    5. **Translate**: Click "Translate Text" for fast Google Translate
    6. **Summarize**: Use Smart Summary (free) or OpenAI Summary (requires API key)
    7. **Download**: Save your results using the download buttons
    
    **Translation:**
    - **Google Translate**: Fast, reliable, high-quality translation
    - Automatic text chunking for long documents
    - Progress indicators for processing
    
    **Summary Options:**
    - **Smart Summary**: Fast, rule-based intelligent extraction (FREE!)
    - **AI Summary**: Advanced analysis using OpenAI GPT (requires API key)
    
    **Performance Optimized:**
    - Fast Google Translate (no slow transformer models)
    - Efficient text processing for long meetings
    - Smart sentence extraction for summaries
    - Multi-language summary templates
    
    **Supported Languages:** English, Ukrainian, Russian, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi
    """)