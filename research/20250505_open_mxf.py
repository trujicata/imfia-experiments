# %%
import start  # noqa

import random

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np

# %%
mxf_video_path = "data/Clip0064.MXF"

probe = ffmpeg.probe(mxf_video_path)
video_stream = next(
    (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
)
width = int(video_stream["width"])
height = int(video_stream["height"])


# %%
def plot_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def sample_random_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frames.append(frame)
    cap.release()
    return frames


def plot_frames(frames):
    fig, axes = plt.subplots(len(frames), 1, figsize=(20, 20))
    for ax, frame in zip(axes, frames):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    plt.show()


# %%
frames = sample_random_frames(mxf_video_path, num_frames=2)
plot_frames(frames)
# %%
frames

# %%
frame = frames[0]
# If it’s too dark, boost contrast/brightness:
frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
plot_frame(frame)
# %%
blur = cv2.GaussianBlur(frame, (5, 5), 0)
plot_frame(blur)

# %%
# 2. Convert to HSV
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
plot_frame(hsv)

# %%
# 3. Threshold for “green”
#    Hue for green is roughly 40–80; Sat & Val thresholds depend on your scene
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
plot_frame(mask)

# %%
# 4. Clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

plot_frame(mask)
# %%
# 5. Extract the green stream
result = cv2.bitwise_and(frame, frame, mask=mask)
plot_frame(result)

# %%
# 6. (Optional) Find biggest contour if you want just the main stream
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # pick the largest by area
    c = max(contours, key=cv2.contourArea)
    stream_mask = np.zeros_like(mask)
    cv2.drawContours(stream_mask, [c], -1, 255, cv2.FILLED)
    result = cv2.bitwise_and(frame, frame, mask=stream_mask)

    plot_frame(result)

# %%
