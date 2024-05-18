import cv2
import numpy as np
from tqdm import tqdm

def brightness_to_speed(brightness):
    min_speed, max_speed = 0, 70
    return (brightness / 255.0) * (max_speed - min_speed) + min_speed

def generate_white_noise(frame_height, frame_width):
    return np.random.randint(0, 256, (frame_height, frame_width), dtype=np.uintp)

def process_video(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Progress bar setup
    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frames")

    base_noise = generate_white_noise(frame_height, frame_width)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate speed factor for each pixel
        speeds = brightness_to_speed(gray_frame)

        # Create noise frame by adjusting base noise with speeds
        noise_frame = (base_noise * speeds).astype(np.uint8)

        # Write the noise frame to the output video
        out.write(noise_frame)
        progress_bar.update(1)

    # Release everything if job is finished
    cap.release()
    out.release()
    progress_bar.close()
    print("Done processing video.")

# Example usage
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'
process_video(input_video_path, output_video_path)
