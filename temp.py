from PIL import Image
import numpy as np
from palette import *
from util import *
from transfer import *
import cv2

# def color_transfer_between_images(source_img_path, style_img_path, k=5):
#     sample_level = 16
#     sample_colors = sample_RGB_color(sample_level)

#     source_img = Image.open(source_img_path)
#     style_img = Image.open(style_img_path)

#     source_img_lab = rgb2lab(source_img)
#     style_img_lab = rgb2lab(style_img)

#     style_colors = style_img_lab.getdata()
#     bins = {}
#     for pixel in style_colors:
#         bins[pixel] = bins.get(pixel, 0) + 1
#     bins = sample_bins(bins)
#     style_means, _ = k_means(bins, k=k, init_mean=True)

#     source_colors = source_img_lab.getdata()
#     bins = {}
#     for pixel in source_colors:
#         bins[pixel] = bins.get(pixel, 0) + 1
#     bins = sample_bins(bins)
#     source_means, _ = k_means(bins, k=k, init_mean=True)

#     sample_weight_map = rbf_weights(source_means, sample_colors)

#     transferred_img_rgb = img_color_transfer(
#         source_img_lab, source_means, style_means,
#         sample_weight_map, sample_colors, sample_level
#     )

#     return transferred_img_rgb

# if __name__ == '__main__':
#     source_img_path = 'output/myimage.jpg'
#     style_img_path = 'output/Lenna.png'

#     transferred_image = color_transfer_between_images(source_img_path, style_img_path)
#     transferred_image.show()  # Show the result

def color_transfer_between_images(source_img, style_img, k=5):
    sample_level = 16
    sample_colors = sample_RGB_color(sample_level)

    source_img_lab = rgb2lab(source_img)
    style_img_lab = rgb2lab(style_img)

    style_colors = style_img_lab.getdata()
    bins = {}
    for pixel in style_colors:
        bins[pixel] = bins.get(pixel, 0) + 1
    bins = sample_bins(bins)
    style_means, _ = k_means(bins, k=k, init_mean=True)

    source_colors = source_img_lab.getdata()
    bins = {}
    for pixel in source_colors:
        bins[pixel] = bins.get(pixel, 0) + 1
    bins = sample_bins(bins)
    source_means, _ = k_means(bins, k=k, init_mean=True)

    sample_weight_map = rbf_weights(source_means, sample_colors)

    transferred_img_rgb = img_color_transfer(
        source_img_lab, source_means, style_means,
        sample_weight_map, sample_colors, sample_level
    )
    
    return transferred_img_rgb


def process_video(source_video_path, style_img_path, output_video_path, k=5):
    # Open the video file
    cap = cv2.VideoCapture(source_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Load style image and convert to PIL format
    style_img = Image.open(style_img_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply color transfer
        transferred_frame_rgb = color_transfer_between_images(frame_pil, style_img, k)
        
        # Convert back to OpenCV format and write to output video
        transferred_frame_bgr = cv2.cvtColor(np.array(transferred_frame_rgb), cv2.COLOR_RGB2BGR)
        out.write(transferred_frame_bgr)
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source_video_path = 'Venice.mp4'
    style_img_path = 'output/JohnWIck.jpg'
    output_video_path = 'output/processed_video_johnwick.avi'

    process_video(source_video_path, style_img_path, output_video_path)