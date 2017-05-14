from mask_generator import *
from perspective_transformer import PerspectiveTransformer
from Camera import Camera
from utils.lane_utils import *
from line import Line


class LaneFinder:
    def __init__(self):
        """
        Find lanes on images using:
            - Sobel operations
            - Color extraction (white and yellow)
            - Sliding histogram
        """
        self.camera_calibrator = Camera()
        self.perspective_transformer = PerspectiveTransformer()

        self.n_frames = 7  # window for averaging

        self.line_segments = 10
        self.image_offset = 250

        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.car_offset = 0.0

        self.dists = []

    def render_hud(self, img):
        """
        Display HUD with information like lane curvature and car's lane center deviance

        :param img: input frame to render lanes on top of
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = "Radius of curvature is {:.2f}m".format(self.curvature)
        cv2.putText(img, text, (50, 50), font, 1, (255, 255, 255), 2)

        left_or_right = 'left' if self.car_offset < 0 else 'right'
        text = "Car is {:.2f}m {} of center".format(np.abs(self.car_offset), left_or_right)
        cv2.putText(img, text, (50, 100), font, 1, (255, 255, 255), 2)

    def render_predicted_lane_area(self, img):
        """
        Render predicted lane area onto an image

        :param img: input frame to render lanes on top of
        """
        overlay = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        mask = np.zeros([img.shape[0], img.shape[1]])

        # lane area
        lane_area = calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = self.perspective_transformer.inverse_transform(mask)

        overlay[mask == 1] = (128, 255, 0)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.5 + overlay[selection] * 0.5

        # side lines
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.best_fit_poly, 5, 255) # draw left lane
        mask = draw_poly(mask, self.right_line.best_fit_poly, 5, 255) # draw right lane
        mask = draw_poly(mask, self.center_poly, 5, 255) # draw center lane
        mask = self.perspective_transformer.inverse_transform(mask)
        img[mask == 255] = (255, 200, 2)

    def debug_frame(self, frame):
        """
        Apply perspective transformation and lane thresholding on an image

        :param frame: input frame
        :return: image used for debugging
        """
        # undistort frame
        frame = self.camera_calibrator.undistort(frame)

        # apply sobel and color transforms to create a thresholded binary image.
        frame = generate_lane_mask(frame, 400)

        # apply perspective transform to get birds-eye view
        frame = self.perspective_transformer.transform(frame)

        frame[frame == 1] = 255

        frame = np.dstack((frame, frame, frame))
        return frame

    def process_frame(self, frame):
        """
        Apply lane detection on a single image

        :param frame: input frame
        :return: processed frame
        """
        orig_frame = np.copy(frame)

        # undistort frame
        frame = self.camera_calibrator.undistort(frame)

        # apply sobel and color transforms to create a binary lane mask
        frame = generate_lane_mask(frame, 400)

        # apply perspective transform to get birds-eye view of road
        frame = self.perspective_transformer.transform(frame)

        left_detected = False
        right_detected = False
        left_x = []
        left_y = []
        right_x = []
        right_y = []

        # if lanes not detected, search around the area where the lanes where last known
        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = detect_lane_along_poly(frame, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detect_lane_along_poly(frame, self.right_line.best_fit_poly, self.line_segments)

            left_detected, right_detected = self.validate_lines(left_x, left_y, right_x, right_y)

        # if left lane hasn't been found, perform a full search of the image with a histogram
        if not left_detected:
            left_x, left_y = histogram_lane_detection(
                frame,
                self.line_segments,
                (self.image_offset, frame.shape[1] // 2),
                h_window=7
            )

            left_x, left_y = outlier_removal(left_x, left_y)  # remove weird horizontal outliers in the detection

        # if right lane hasn't been found, perform a full search of the image with a histogram
        if not right_detected:
            right_x, right_y = histogram_lane_detection(
                frame,
                self.line_segments,
                (frame.shape[1] // 2, frame.shape[1] - self.image_offset),
                h_window=7
            )

            right_x, right_y = outlier_removal(right_x, right_y)

        if not left_detected or not right_detected:
            left_detected, right_detected = self.validate_lines(left_x, left_y, right_x, right_y)

        # updated left lane
        if left_detected:

            # switch x and y since lines are basically vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.n_frames, left_y, left_x)

        # updated right lane
        if right_detected:

            # switch x and y since lines are basically vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.n_frames, right_y, right_x)

        # if both lanes are found, then calculate curvature, center offset and a center-lane polynomial
        if self.left_line is not None and self.right_line is not None:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = calc_curvature(self.center_poly)
            self.car_offset = (frame.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700

            self.render_predicted_lane_area(orig_frame)
            self.render_hud(orig_frame)

        return orig_frame

    def validate_lines(self, left_x, left_y, right_x, right_y):
        """
        Compare two lines to each other and to their last prediction

        :param left_x: predicted x coordinates for left lane
        :param left_y: predicted y coordinates for left lane
        :param right_x: predicted x coordinates for right lane
        :param right_y: predicted y coordinates for right lane
        :return: boolean tuple (left_is_good, right_is_good)
        """
        left_is_good = False
        right_is_good = False

        if self.is_line_plausible((left_x, left_y), (right_x, right_y)):
            left_is_good = True
            right_is_good = True

        elif self.left_line is not None and self.right_line is not None:

            if self.is_line_plausible((left_x, left_y), (self.left_line.all_y, self.left_line.all_x)):
                left_is_good = True

            if self.is_line_plausible((right_x, right_y), (self.right_line.all_y, self.right_line.all_x)):
                right_is_good = True

        return left_is_good, right_is_good

    def is_line_plausible(self, left, right):
        """
        Determine if the detected pixels describing two lines are plausible lane lines based on curvature and distance.
        :param left: Tuple of arrays containing the coordinates of detected lane pixels
        :param right: Tuple of arrays containing the coordinates of detected lane pixels
        :return:
        """

        # ignore lines that have less then 3 points
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False

        # prepare some temporary line objects and test if they're plausible
        new_left = Line(y=left[0], x=left[1])
        new_right = Line(y=right[0], x=right[1])
        return are_lanes_plausible(new_left, new_right)


if __name__ == '__main__':

    lane_finder = LaneFinder()

    images_left = []
    images_right = []

    test_images = ["./test_images/straight_lines2.jpg", "./test_images/test5.jpg"]

    for image in test_images:
        image = imread(image)
        transformed = lane_finder.process_frame(image)
        images_left.append(image)
        images_right.append(transformed)

    plot_side_by_side_images(images_left, images_right, 'Original', 'Lane Found', './output_images/lane_found.png')

    print('Done')
