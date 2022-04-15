
import cv2
import numpy as np
# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}
p_color = [(0, 255, 255),
           (0, 191, 255),
           (0, 255, 102),
           (0, 77, 255),
           (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255),
           (77, 255, 204),
           (77, 204, 255),
           (191, 255, 77),
           (77, 191, 255),
           (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255),
           (77, 255, 204),
           (191, 77, 255),
           (77, 255, 191),
           (127, 77, 255),
           (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (0, 215, 255), # head
    (0, 2): (0, 255, 204),
    (1, 3): (0, 134, 255),
    (2, 4): (0, 255, 50),

    (5, 6): (77, 255, 222),  # top limbs
    (5, 7): (77, 196, 255),
    (7, 9): (77, 135, 255),
    (6, 8): (191, 255, 77),
    (8, 10): (77, 255, 77),

    (17, 11): (77, 222, 255),
    (17, 12): (255, 156, 127),   # Body

    (11, 13): (0, 127, 255), # bottom limbs
    (12, 14): (255, 127, 77),
    (13, 15): (0, 77, 255),
    (14, 16): (255, 77, 36)
}


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def keypoints(img, keypoints, fps):
    h, w, channel = img.shape
    for i, k in enumerate(keypoints[0, 0, :, :]):
        # Converts to numpy array
        k = k.numpy()
        # Checks confidence for keypoint
        if k[2] > 0.35:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            hc = int(k[0] * h)
            wc = int(k[1] * w)
            color = p_color[i]
            # Draws a circle on the image for each keypoint
            img = cv2.circle(img, (wc, hc), 2, color, 2)

    text = "FPS:{}".format(fps)
    draw_text(img, text, pos=(1, 1), font_scale=1, font_thickness=1)
    return img

def skeleton(image, keypoints_with_scores, fps):
    # Draws skeleton on image.
    height, width, channel = image.shape
    # process neck point
    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(keypoints_with_scores, height, width)

    for i, (ptStart, ptEnd, scores) in enumerate(keypoint_edges):
        ptStart = [int(round(x)) for x in ptStart]
        ptEnd = [int(round(x)) for x in ptEnd]
        cv2.line(image, ptStart, ptEnd, edge_colors[i], 2)
        cv2.line(image, ptStart, ptEnd, edge_colors[i], 2 * int(scores[0] + scores[1]) + 1)

    text = "FPS:{}".format(fps)
    draw_text(image, text, pos=(1, 1),font_scale = 1, font_thickness=1)
    # cv2.putText(image, str(fps), (5, 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 0), 1, cv2.LINE_AA)

    return image


def _keypoints_and_edges_for_display(keypoints_with_scores,height, width, keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)
        for i, (edge_pair, color) in enumerate(KEYPOINT_EDGE_INDS_TO_COLOR.items()):
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                score_start = kpts_scores[edge_pair[0]]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                score_end =  kpts_scores[edge_pair[1]]
                line_seg = np.array([[x_start, y_start], [x_end, y_end], [score_start, score_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 18, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors



