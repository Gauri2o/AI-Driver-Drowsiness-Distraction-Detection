import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"
OUTPUT_FILE = "features_combined.csv"

SUBJECTS = ["S01", "S02"]

BEHAVIORS = [
    "attentive",
    "drowsy",
    "yawning"
]

ALLOWED_CONDITIONS = [
    "normal",
    "glasses",
    "mask"
]

SKIP_CONDITIONS = [
    "low_light"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

FRAME_STEP = 1


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ============================================================
# LANDMARKS
# ============================================================

LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]

MOUTH = [
    61, 291, 13, 14, 78, 308
]

POSE_POINTS = [
    1, 152, 10, 33, 263, 61
]


# ============================================================
# EAR
# ============================================================

def calculate_ear(points):

    if len(points) != 6:
        return 0.0

    A = distance.euclidean(
        points[1],
        points[5]
    )

    B = distance.euclidean(
        points[2],
        points[4]
    )

    C = distance.euclidean(
        points[0],
        points[3]
    )

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


# ============================================================
# MAR
# ============================================================

def calculate_mar(points):

    if len(points) != 6:
        return 0.0

    A = distance.euclidean(
        points[2],
        points[3]
    )

    B = distance.euclidean(
        points[4],
        points[5]
    )

    C = distance.euclidean(
        points[0],
        points[1]
    )

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


# ============================================================
# BRIGHTNESS
# ============================================================

def calculate_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(np.mean(gray))


# ============================================================
# JAW RATIO
# ============================================================

def calculate_jaw_ratio(face_landmarks):

    nose = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    forehead = face_landmarks.landmark[10]

    face_height = abs(
        chin.y - forehead.y
    )

    if face_height < 0.001:
        return 0.0

    jaw_ratio = (
        chin.y - nose.y
    ) / face_height

    return float(jaw_ratio)


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(
    face_landmarks,
    width,
    height
):

    image_points = []

    for idx in POSE_POINTS:

        landmark = face_landmarks.landmark[idx]

        x = landmark.x * width
        y = landmark.y * height

        image_points.append(
            (x, y)
        )

    image_points = np.array(
        image_points,
        dtype=np.float64
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (0.0, 330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (0.0, -150.0, -125.0)
        ],
        dtype=np.float64
    )

    focal_length = width

    center = (
        width / 2,
        height / 2
    )

    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    dist_coeffs = np.zeros(
        (4, 1),
        dtype=np.float64
    )

    try:

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return yaw, pitch, roll

    except Exception:

        return 0.0, 0.0, 0.0


# ============================================================
# EXTRACT FEATURES FROM ONE IMAGE
# ============================================================

def extract_features(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None

    height, width = image.shape[:2]

    brightness = calculate_brightness(
        image
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb
    )

    # --------------------------------------------------------
    # FACE NOT DETECTED
    # --------------------------------------------------------

    if not results.multi_face_landmarks:

        return {
            "face_detected": 0,
            "ear": 0.0,
            "mar": 0.0,
            "jaw_ratio": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "brightness": round(
                brightness,
                2
            )
        }

    face_landmarks = (
        results.multi_face_landmarks[0]
    )

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_eye_points = []

    for idx in LEFT_EYE:

        landmark = face_landmarks.landmark[idx]

        x = landmark.x * width
        y = landmark.y * height

        left_eye_points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    right_eye_points = []

    for idx in RIGHT_EYE:

        landmark = face_landmarks.landmark[idx]

        x = landmark.x * width
        y = landmark.y * height

        right_eye_points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # MOUTH
    # --------------------------------------------------------

    mouth_points = []

    for idx in MOUTH:

        landmark = face_landmarks.landmark[idx]

        x = landmark.x * width
        y = landmark.y * height

        mouth_points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # EAR
    # --------------------------------------------------------

    left_ear = calculate_ear(
        left_eye_points
    )

    right_ear = calculate_ear(
        right_eye_points
    )

    ear = (
        left_ear +
        right_ear
    ) / 2.0

    # --------------------------------------------------------
    # MAR
    # --------------------------------------------------------

    mar = calculate_mar(
        mouth_points
    )

    # --------------------------------------------------------
    # JAW
    # --------------------------------------------------------

    jaw_ratio = calculate_jaw_ratio(
        face_landmarks
    )

    # --------------------------------------------------------
    # HEAD POSE
    # --------------------------------------------------------

    yaw, pitch, roll = calculate_head_pose(
        face_landmarks,
        width,
        height
    )

    return {
        "face_detected": 1,
        "ear": round(
            float(ear),
            5
        ),
        "mar": round(
            float(mar),
            5
        ),
        "jaw_ratio": round(
            float(jaw_ratio),
            5
        ),
        "yaw": round(
            float(yaw),
            3
        ),
        "pitch": round(
            float(pitch),
            3
        ),
        "roll": round(
            float(roll),
            3
        ),
        "brightness": round(
            float(brightness),
            2
        )
    }


# ============================================================
# FIND DATASET IMAGES
# ============================================================

def find_images():

    records = []

    for subject in SUBJECTS:

        subject_dir = os.path.join(
            DATASET_DIR,
            subject
        )

        if not os.path.isdir(subject_dir):

            print(
                f"WARNING: {subject_dir} not found"
            )

            continue

        # --------------------------------------------------------
        # Only process these condition folders
        # --------------------------------------------------------

        for condition in ALLOWED_CONDITIONS:

            condition_dir = os.path.join(
                subject_dir,
                condition
            )

            if not os.path.isdir(condition_dir):

                continue

            for behavior in BEHAVIORS:

                behavior_dir = os.path.join(
                    condition_dir,
                    behavior
                )

                if not os.path.isdir(behavior_dir):

                    continue

                for filename in os.listdir(
                    behavior_dir
                ):

                    if not filename.lower().endswith(
                        IMAGE_EXTENSIONS
                    ):
                        continue

                    full_path = os.path.join(
                        behavior_dir,
                        filename
                    )

                    relative_path = os.path.relpath(
                        full_path,
                        DATASET_DIR
                    ).replace(
                        os.sep,
                        "/"
                    )

                    records.append(
                        {
                            "image_path": relative_path,
                            "subject_id": subject,
                            "behavior": behavior,
                            "condition": condition,
                            "lighting": condition
                        }
                    )

    records.sort(
        key=lambda x: x["image_path"]
    )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print(" COMBINED DRIVER DATASET FEATURE EXTRACTION")
    print("=" * 65)
    print()

    if not os.path.isdir(
        DATASET_DIR
    ):

        print(
            "ERROR: dataset folder not found."
        )

        return

    records = find_images()

    if not records:

        print(
            "ERROR: No dataset images found."
        )

        return

    print(
        f"Images found: {len(records)}"
    )

    print()
    print(
        "Subjects:"
    )

    for subject in SUBJECTS:

        count = sum(
            1
            for r in records
            if r["subject_id"] == subject
        )

        print(
            f"  {subject}: {count}"
        )

    print()
    print(
        "Low-light images: SKIPPED"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()

    fieldnames = [
        "image_path",
        "subject_id",
        "behavior",
        "condition",
        "lighting",
        "timestamp",

        "face_detected",

        "ear",
        "mar",
        "jaw_ratio",

        "yaw",
        "pitch",
        "roll",

        "brightness"
    ]

    processed = 0
    successful = 0
    failed = 0
    face_not_detected = 0

    # --------------------------------------------------------
    # WRITE TO NEW FILE
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, row in enumerate(
            records
        ):

            if index % FRAME_STEP != 0:
                continue

            processed += 1

            image_path = os.path.join(
                DATASET_DIR,
                row["image_path"].replace(
                    "/",
                    os.sep
                )
            )

            try:

                if not os.path.exists(
                    image_path
                ):

                    failed += 1

                    print(
                        f"Missing: {image_path}"
                    )

                    continue

                features = extract_features(
                    image_path
                )

                if features is None:

                    failed += 1

                    continue

                output_row = {

                    "image_path":
                        row["image_path"],

                    "subject_id":
                        row["subject_id"],

                    "behavior":
                        row["behavior"],

                    "condition":
                        row["condition"],

                    "lighting":
                        row["lighting"],

                    "timestamp":
                        "",

                    "face_detected":
                        features["face_detected"],

                    "ear":
                        features["ear"],

                    "mar":
                        features["mar"],

                    "jaw_ratio":
                        features["jaw_ratio"],

                    "yaw":
                        features["yaw"],

                    "pitch":
                        features["pitch"],

                    "roll":
                        features["roll"],

                    "brightness":
                        features["brightness"]
                }

                writer.writerow(
                    output_row
                )

                successful += 1

                if features[
                    "face_detected"
                ] == 0:

                    face_not_detected += 1

                if processed % 50 == 0:

                    print(
                        f"Processed: {processed} | "
                        f"Successful: {successful} | "
                        f"Failed: {failed} | "
                        f"Face missed: {face_not_detected}"
                    )

            except Exception as error:

                failed += 1

                print()
                print(
                    f"Error processing: "
                    f"{row['image_path']}"
                )

                print(
                    f"Reason: {error}"
                )

    face_mesh.close()

    print()
    print("=" * 65)
    print(" FEATURE EXTRACTION COMPLETED")
    print("=" * 65)
    print()

    print(
        f"Total records     : {processed}"
    )

    print(
        f"Successful        : {successful}"
    )

    print(
        f"Failed            : {failed}"
    )

    print(
        f"Face not detected : {face_not_detected}"
    )

    print()

    print(
        f"Output file       : {OUTPUT_FILE}"
    )

    print()
    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()