import numpy as np


class Line:
    def __init__(self, n_frames=1, x=None, y=None):
        """
        Track the characteristics of an independent lane line
        :param n_frames: Number of frames for smoothing
        :param x: initial x coordinates
        :param y: initial y coordinates
        """
        # Frame memory
        self.n_frames = n_frames

        # was the line detected in the last iteration?
        self.detected = False

        # number of pixels added per frame
        self.n_pixel_per_frame = []

        # x values of the last n fits of the line
        self.recent_x_fitted = []

        # average x values of the fitted line over the last n iterations
        self.best_x = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = None

        # Polynom for the current coefficients
        self.current_fit_poly = None

        # Polynom for the average coefficients over the last n iterations
        self.best_fit_poly = None

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.all_x = None

        # y values for detected line pixels
        self.all_y = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        """
        Update the lane parameters.
        :param x: A list of x coordinates
        :param y: A list of y coordinates
        """
        self.all_x = x
        self.all_y = y

        self.n_pixel_per_frame.append(len(self.all_x))
        self.recent_x_fitted.extend(self.all_x)

        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_x_fitted = self.recent_x_fitted[n_x_to_remove:]

        self.best_x = np.mean(self.recent_x_fitted)

        self.current_fit = np.polyfit(self.all_x, self.all_y, 2)

        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def is_current_fit_parallel(self, other_line, threshold=(0, 0)):
        """
        Check if two lines are parallel by comparing their first two coefficients (Bx + C).
        :param other_line: Line object to compare against
        :param threshold: Tuple of float values representing the delta thresholds for the coefficients
        :return:
        """
        first_coefi_dif = np.abs(self.current_fit[0] - other_line.current_fit[0])
        second_coefi_dif = np.abs(self.current_fit[1] - other_line.current_fit[1])

        is_parallel = first_coefi_dif < threshold[0] and second_coefi_dif < threshold[1]

        return is_parallel

    def get_current_fit_distance(self, other_line):
        """
        Get the instantaneous distance between another lane polynomial
        :param other_line: Line object with a lane polyfit
        :return: Instantaneous distance between two polynomials
        """
        return np.abs(self.current_fit_poly(719) - other_line.current_fit_poly(719))

    def get_best_fit_distance(self, other_line):
        """
        Get the distance between the best fit polynomials of two Line objects
        :param other_line: Line object with a lane polyfit
        :return: Distance between another polynomial
        """
        return np.abs(self.best_fit_poly(719) - other_line.best_fit_poly(719))
