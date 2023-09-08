import speech_recognition as sr
from transformers import pipeline
from pytube import YouTube
from moviepy.editor import VideoFileClip

# Load the speech recognition model
r = sr.Recognizer()

# Load the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

# Specify the video link
video_link = "https://www.youtube.com/watch?v=OP9-kOCSqY8"  # replace with the actual video link

try:
    # Download the video and extract the audio
    yt = YouTube(video_link)
    audio = yt.streams.filter(only_audio=True).first().download()

    # Convert audio format if needed (optional)
    audio_wav = audio.split(".mp4")[0] + ".wav"
    if audio.endswith(".mp4"):
        clip = VideoFileClip(audio)
        clip.audio.write_audiofile(audio_wav)
    
    # Load the audio file
    with sr.AudioFile(audio_wav) as source:
        audio = r.record(source)

    # Transcribe the audio to text
    text = r.recognize_google(audio)

    # Summarize the text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)

    print(summary[0]['summary_text'])

except Exception as e:
    print("Error:", str(e))
