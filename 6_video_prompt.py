import shutil
import cv2
import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


# Create or cleanup existing extracted image frames directory.
FRAME_EXTRACTION_DIRECTORY = "data/videos/frames"
FRAME_PREFIX = "_frame"


class File:
    def __init__(self, file_path: str, display_name: str = None):
        self.file_path = file_path
        if display_name:
            self.display_name = display_name
        self.timestamp = self._get_timestamp(file_path)

    def set_file_metadata(self, metadata):
        self.metadata = metadata

    def _get_timestamp(self, filename):
        """Extracts the frame count (as an integer) from a filename with the format
        'output_file_prefix_frame00:00.jpg'.
        """
        parts = filename.split(FRAME_PREFIX)
        if len(parts) != 2:
            return None  # Indicates the filename might be incorrectly formatted
        return parts[1].split('.')[0]


def create_frame_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


# Reference https://github.com/google-gemini/cookbook/blob/main/quickstarts/Video.ipynb
def extract_frame_from_video(video_file_path):
    print(f"""Extracting {
          video_file_path} at 1 frame per second. This might take a bit...""")
    create_frame_output_dir(FRAME_EXTRACTION_DIRECTORY)
    vidcap = cv2.VideoCapture(video_file_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    output_file_prefix = os.path.basename(video_file_path).replace('.', '_')
    frame_count = 0
    count = 0
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:  # End of video
            break
        if int(count / fps) == frame_count:  # Extract a frame every second
            min = frame_count // 60
            sec = frame_count % 60
            time_string = f"{min:02d}:{sec:02d}"
            image_name = f"{output_file_prefix}{FRAME_PREFIX}{time_string}.jpg"
            output_filename = os.path.join(
                FRAME_EXTRACTION_DIRECTORY, image_name)
            cv2.imwrite(output_filename, frame)
            frame_count += 1
        count += 1
    vidcap.release()  # Release the capture object\n",
    print(f"""Completed video frame extraction!\n\nExtracted: {
          frame_count} frames""")


def upload_files():
    # Process each frame in the output directory
    files = os.listdir(FRAME_EXTRACTION_DIRECTORY)
    files = sorted(files)
    files_to_upload = []
    for file in files:
        files_to_upload.append(
            File(file_path=os.path.join(FRAME_EXTRACTION_DIRECTORY, file)))

    # Upload the files to the API
    # Only upload a 10 second slice of files to reduce upload time.
    # Change full_video to True to upload the whole video.
    full_video = False

    uploaded_files = []
    print(f"""Uploading {len(files_to_upload)
                         if full_video else 10} files. This might take a bit...""")

    for file in files_to_upload if full_video else files_to_upload[40:50]:
        print(f'Uploading: {file.file_path}...')
        file_metadata = genai.upload_file(path=file.file_path)
        file.set_file_metadata(file_metadata)
        uploaded_files.append(file)

    print(f"Completed file uploads!\n\nUploaded: {len(uploaded_files)} files")
    return uploaded_files


def make_prompt(prompt, files):
    request = [prompt]
    for file in files:
        request.append(file.timestamp)
        request.append(file.metadata)
    return request


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    extract_frame_from_video("data/videos/video_example.mp4")
    files = upload_files()

    prompt = make_prompt(
        prompt="Provide a brief summary over the video:", files=files)

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        console.print(chunk.text)
