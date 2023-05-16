import argparse
import os
import re
import string
import warnings

import isodate
import whisper
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pydub import AudioSegment
from pytube import YouTube

# Your API key goes here
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# set up prompts here for formatting
map_prompt_template = """
The following is the transcript of a video. Please provide a brief summary of the video, including the main points and key takeaways. Output should be as a markdown outline.

{text}

BRIEF SUMMARY IN MARKDOWN FORMAT:
"""

MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """Here are a few markdown outlines of the video. Please combine them into a single outline.

{text}

COMBINED OUTLINE IN MARKDOWN FORMAT:
"""

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["text"]
)


def create_summary_filename(video_title, channel_title):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    safe_video_title = (
        "".join(c for c in video_title if c in valid_chars).strip().replace(" ", "_")
    )
    safe_channel_title = (
        "".join(c for c in channel_title if c in valid_chars).strip().replace(" ", "_")
    )
    filename = f"summaries/{safe_channel_title}_{safe_video_title}.md"
    return filename


# Get video ID from the URL
def get_video_id(url):
    video_id = None
    pattern = re.compile(
        r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)"
    )
    match = pattern.match(url)
    if match:
        video_id = match.group(4)
    return video_id


# Get video details using YouTube Data API v3
def get_video_details(video_id):
    try:
        youtube = build("youtube", "v3", developerKey=GOOGLE_API_KEY)
        response = (
            youtube.videos().list(part="snippet,contentDetails", id=video_id).execute()
        )

        return response["items"][0] if response["items"] else None
    except HttpError as e:
        print(f"An error occurred: {e}")
        return None


# Convert ISO 8601 duration to a human-readable format
def format_duration(duration):
    parsed_duration = isodate.parse_duration(duration)
    total_seconds = int(parsed_duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate_unique_filename(video_id, prefix, extension):
    return f"{prefix}/{video_id}.{extension}"


def transcribe_audio(video_id, video_url):
    # Create directories for audio_streams and transcriptions if they don't exist
    os.makedirs("audio_streams", exist_ok=True)
    os.makedirs("transcriptions", exist_ok=True)

    transcription_filename = generate_unique_filename(video_id, "transcriptions", "txt")

    if os.path.exists(transcription_filename):
        with open(transcription_filename, "r") as transcription_file:
            transcription = transcription_file.read()
    else:
        # Download the video as audio
        yt = YouTube(video_url)
        video = yt.streams.filter(only_audio=True).first()
        audio_filename = generate_unique_filename(video_id, "audio_streams", "mp4")
        file_name = video.download(filename=audio_filename)

        # Convert the audio file to WAV format
        audio = AudioSegment.from_file(file_name)
        audio.export("audio.wav", format="wav")

        # Load the Whisper ASR model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = whisper.load_model("base")

        # Transcribe the audio
        result = model.transcribe("audio.wav")
        transcription = result["text"]

        # Save the transcription to a file
        with open(transcription_filename, "w") as transcription_file:
            transcription_file.write(transcription)

        # Cleanup
        os.remove("audio.wav")

    return transcription


def split_text_to_documents(text, max_length=4096, overlap=100):
    tokens = text.split()
    text_chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        if current_length + len(token) + 1 > max_length - overlap:
            text_chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = sum(len(t) + 1 for t in current_chunk)

        current_chunk.append(token)
        current_length += len(token) + 1

    if current_chunk:
        text_chunks.append(" ".join(current_chunk))

    return [Document(page_content=t) for t in text_chunks]


# Main function
def main(args):
    if args.url:
        url = args.url
    else:
        url = input("Please enter a YouTube video URL: ")

    video_id = get_video_id(url)
    if not video_id:
        print("Invalid YouTube URL")
        return

    embed_url = f"https://www.youtube.com/embed/{video_id}"
    embed_code = f'<iframe width="560" height="315" src="{embed_url}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'

    video_details = get_video_details(video_id)
    if not video_details:
        print("Could not fetch video details")
        return

    snippet = video_details["snippet"]
    content_details = video_details["contentDetails"]

    title = snippet["title"]
    description = snippet["description"]
    channel_title = snippet["channelTitle"]
    length = format_duration(content_details["duration"])
    published_at = snippet["publishedAt"]

    markdown_block = f"""
{embed_code}
## {title}

**Channel**: {channel_title}

**Published**: {published_at}

**Length**: {length}

**Description**:
{description}
"""

    print(markdown_block)

    if args.transcribe:
        transcription = transcribe_audio(video_id, url)
        if args.summary:
            llm = OpenAI(temperature=0)

            # Split the transcription into smaller chunks as Documents
            docs = split_text_to_documents(transcription)

            # Choose a chain type for summarization
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=MAP_PROMPT,
                combine_prompt=COMBINE_PROMPT,
            )

            # Run the summarization chain
            summary = chain.run(docs)

            # Print or store the summary
            print(summary)

            output_filename = create_summary_filename(title, channel_title)

            os.makedirs("summaries", exist_ok=True)

            with open(output_filename, "w") as output_file:
                output_file.write(markdown_block)

                if args.transcribe:
                    transcription = transcribe_audio(video_id, url)
                    if args.summary:
                        output_file.write(f"\n\n{summary}")

            print(f"Summary saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch YouTube video information and generate a markdown block"
    )
    parser.add_argument("-u", "--url", help="YouTube video URL")
    parser.add_argument(
        "-t", "--transcribe", action="store_true", help="Transcribe the video audio"
    )
    parser.add_argument(
        "-s", "--summary", action="store_true", help="Summarize the video transcript"
    )
    args = parser.parse_args()
    main(args)
