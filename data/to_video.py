import os 
import glob 
import cv2 


def create_video(image_folder, video_name, interval=5, fps=30):
    # Get al image file paths
    images = glob.glob(f"{image_folder}/*.jpg")

    images.sort()

    sampled_images = images[::interval][:500]

    frame = cv2.imread(sampled_images[0])
    height, width, layers = frame.shape

    fource = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fource, fps, (width, height))
    
    for image in sampled_images:
        video.write(cv2.imread(image))
    
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    path_to_images = "/data/leuven/333/vsc33366/projects/Diffusion-Video/dataset/Lasot/bear/bear-1/img"
    create_video(image_folder=path_to_images,
                 video_name='./output_video.mp4',
                 interval=10,
                 fps=8)