from moviepy.video.io.VideoFileClip import VideoFileClip
from lane_finder import LaneFinder

easy = "./videos/project_video.mp4"
medium = "./videos/challenge_video.mp4"
hard = "./videos/harder_challenge_video.mp4"

input_video = hard
debug = False  # output a debugging version of the video

if __name__ == '__main__':
    """
    Process each frame of a video to detect lanes and render output video
    """
    lane_finder = LaneFinder()

    video_clip = VideoFileClip(input_video)

    if debug:
        processed_clip = video_clip.fl_image(lane_finder.debug_frame)
    else:
        processed_clip = video_clip.fl_image(lane_finder.process_frame)

    # save video
    processed_clip.write_videofile(input_video[:-4] + '_processed.mp4', audio=False)

    print('Done')
