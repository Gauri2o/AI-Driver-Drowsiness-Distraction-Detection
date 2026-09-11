import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
import winsound
import time
import threading
from collections import deque

from flask import Flask, jsonify, render_template

# =========================================================
# AI MODEL INTEGRATION - ADDITIONAL LAYER ONLY
# =========================================================
try:
    import joblib
    import pandas as pd
except ImportError:
    joblib = None
    pd = None



# =========================================================
# FLASK DASHBOARD
# =========================================================

app = Flask(__name__)

detection_data = {
    "status": "ATTENTIVE",
    "risk_level": "LOW",

    "ear": 0.0,
    "mar": 0.0,
    "jaw_ratio": 0.0,

    "head_position": "ATTENTIVE",

    "drowsiness": False,
    "yawning": False,
    "distraction": False,

    "lighting": "NORMAL",
    "brightness": 0.0,
    "face": False,

    "glasses": "SUPPORTED",
    "mask": "SUPPORTED",

    "alert": False,

    "fatigue_score": 0,
    "fatigue_level": "LOW",

    "yawn_count": 0,
    "drowsiness_count": 0,
    "distraction_count": 0,

    "session_time": "00:00",

    "ai_prediction": "UNAVAILABLE",
    "ai_confidence": 0.0,
    "ai_model": "NOT LOADED"
}


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():
    return jsonify(detection_data)

@app.route("/api/ml-results")
def ml_results():

    return jsonify({
        "accuracy": 91.81,
        "precision": 91.80,
        "recall": 91.81,
        "f1": 91.80,

        "features": [
            "MAR",
            "EAR",
            "Jaw Ratio",
            "Brightness"
        ],

        "feature_importance": [
            28.64,
            25.20,
            23.55,
            22.61
        ],

        "classes": [
            "Attentive",
            "Drowsy",
            "Yawning"
        ],

        "confusion_matrix": [
            [339, 19, 20],
            [15, 358, 6],
            [23, 8, 323]
        ],

        "distribution": [
            1889,
            1896,
            1770
        ]
    })


def run_dashboard():

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )


dashboard_thread = threading.Thread(
    target=run_dashboard,
    daemon=True
)

dashboard_thread.start()


# =========================================================
# RANDOM FOREST AI MODEL
# =========================================================

AI_MODEL_PATH = "driver_behavior_model_combined.pkl"

ai_model = None
ai_model_loaded = False

try:
    if joblib is None or pd is None:
        raise ImportError("joblib/pandas is not available")

    ai_model = joblib.load(AI_MODEL_PATH)
    ai_model_loaded = True

    detection_data["ai_model"] = "LOADED"

    print()
    print("AI MODEL LOADED SUCCESSFULLY")
    print(f"Model : {AI_MODEL_PATH}")
    print("Loader: joblib")
    print()

except Exception as e:
    print()
    print("WARNING: AI model could not be loaded.")
    print(f"Reason: {e}")
    print("Running with existing computer-vision detection.")
    print()

    detection_data["ai_model"] = "NOT LOADED"


def predict_ai_behavior(ear, mar, jaw_ratio, brightness):
    if not ai_model_loaded or ai_model is None:
        return "UNAVAILABLE", 0.0

    try:
        features = pd.DataFrame(
            [[
                float(ear),
                float(mar),
                float(jaw_ratio),
                float(brightness)
            ]],
            columns=[
                "ear",
                "mar",
                "jaw_ratio",
                "brightness"
            ]
        )

        prediction = str(
            ai_model.predict(features)[0]
        ).upper()

        confidence = 0.0

        if hasattr(ai_model, "predict_proba"):
            probabilities = ai_model.predict_proba(features)[0]
            confidence = float(np.max(probabilities)) * 100.0

        return prediction, confidence

    except Exception:
        return "UNAVAILABLE", 0.0


# =========================================================
# AI TEMPORAL SMOOTHING
# =========================================================

def smooth_ai_prediction(raw_prediction, raw_confidence):
    global stable_ai_prediction, stable_ai_confidence

    if raw_prediction == "UNAVAILABLE" or raw_confidence < AI_MIN_CONFIDENCE:
        return stable_ai_prediction, stable_ai_confidence

    ai_prediction_history.append(raw_prediction)
    ai_confidence_history.append(float(raw_confidence))

    if not ai_prediction_history:
        return stable_ai_prediction, stable_ai_confidence

    # Majority vote over the recent frames.
    counts = {}
    for prediction in ai_prediction_history:
        counts[prediction] = counts.get(prediction, 0) + 1

    candidate = max(counts, key=counts.get)
    votes = counts[candidate]

    if votes >= AI_CONFIRMATION_VOTES:
        selected_confidences = [
            c for p, c in zip(ai_prediction_history, ai_confidence_history)
            if p == candidate
        ]
        stable_ai_prediction = candidate
        stable_ai_confidence = float(np.mean(selected_confidences)) if selected_confidences else 0.0

    return stable_ai_prediction, stable_ai_confidence


def reset_ai_smoothing():
    global stable_ai_prediction, stable_ai_confidence
    ai_prediction_history.clear()
    ai_confidence_history.clear()
    stable_ai_prediction = "UNAVAILABLE"
    stable_ai_confidence = 0.0


# =========================================================
# MEDIAPIPE
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# LANDMARKS
# =========================================================

LEFT_EYE = [
    33, 160, 158,
    133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387,
    263, 373, 380
]

MOUTH = [
    61, 291, 13,
    14, 78, 308
]

FACE_POINTS = [
    1, 152, 33,
    263, 61, 291
]


# =========================================================
# SETTINGS
# =========================================================

EAR_THRESHOLD = 0.20
EYE_FRAME_LIMIT = 10

# ---------------------------------------------------------
# LIVE AI STABILITY
# ---------------------------------------------------------
AI_HISTORY_LENGTH = 12
AI_MIN_CONFIDENCE = 65.0
AI_CONFIRMATION_VOTES = 7
AI_HIGH_CONFIDENCE = 75.0
AI_YAWN_CONFIDENCE = 80.0

# Adaptive EAR calibration
EAR_CALIBRATION_TIME = 2.0
EAR_MIN_THRESHOLD = 0.16
EAR_MAX_THRESHOLD = 0.24
EAR_CALIBRATION_MIN_SAMPLES = 15

MAR_THRESHOLD = 0.68
YAWN_FRAME_LIMIT = 18

JAW_MOVEMENT_THRESHOLD = 0.105
MASK_YAWN_FRAME_LIMIT = 18

YAW_THRESHOLD = 20
PITCH_THRESHOLD = 20

CALIBRATION_TIME = 2.0
DISTRACTION_TIME = 2.0

# ---------------------------------------------------------
# LIGHTING
# ---------------------------------------------------------

LOW_LIGHT_THRESHOLD = 70
VERY_LOW_LIGHT_THRESHOLD = 45

# ---------------------------------------------------------
# FACE LOSS STABILITY
# ---------------------------------------------------------

MAX_FACE_LOST_FRAMES = 8

# ---------------------------------------------------------
# AUDIO
# ---------------------------------------------------------

ALARM_INTERVAL = 2.0


# =========================================================
# COUNTERS / STATE
# =========================================================

eye_closed_frames = 0
mouth_open_frames = 0
jaw_yawn_frames = 0

last_alarm_time = 0

session_start_time = time.time()

yawn_count = 0
drowsiness_count = 0
distraction_count = 0

previous_yawning = False
previous_drowsiness = False
previous_distraction = False


# =========================================================
# FACE TRACKING STABILITY
# =========================================================

face_lost_frames = 0

last_valid_face_time = time.time()

last_ear = 0.0
last_mar = 0.0

last_head_status = "ATTENTIVE"

last_relative_yaw = 0.0
last_relative_pitch = 0.0
last_roll = 0.0


# =========================================================
# HEAD CALIBRATION
# =========================================================

calibration_yaw = []
calibration_pitch = []

calibration_start_time = time.time()

baseline_yaw = 0.0
baseline_pitch = 0.0

calibrated = False


# =========================================================
# JAW CALIBRATION
# =========================================================

jaw_calibration_values = []

jaw_calibration_start = time.time()

JAW_CALIBRATION_TIME = 2.0

jaw_baseline = None


# =========================================================
# JAW HISTORY
# =========================================================

jaw_history = []

MAX_JAW_HISTORY = 15


# =========================================================
# DISTRACTION
# =========================================================

distraction_start_time = None

distraction_confirmed = False

distraction_direction = "NONE"


# =========================================================
# FATIGUE
# =========================================================

fatigue_score = 0.0

# AI state
ai_prediction = "UNAVAILABLE"
ai_confidence = 0.0

ai_prediction_history = deque(maxlen=AI_HISTORY_LENGTH)
ai_confidence_history = deque(maxlen=AI_HISTORY_LENGTH)
stable_ai_prediction = "UNAVAILABLE"
stable_ai_confidence = 0.0

# Adaptive EAR state
ear_calibration_values = []
ear_calibration_start = time.time()
adaptive_ear_threshold = EAR_THRESHOLD
ear_calibrated = False


# =========================================================
# EAR
# =========================================================

def calculate_ear(points):

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


# =========================================================
# MAR
# =========================================================

def calculate_mar(points):

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


# =========================================================
# LIGHTING ENHANCEMENT
# =========================================================

def enhance_lighting(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    # -----------------------------------------------------
    # VERY LOW LIGHT
    # -----------------------------------------------------

    if brightness < VERY_LOW_LIGHT_THRESHOLD:

        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.2,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel
            )
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        # Mild gamma correction
        gamma = 1.25

        table = np.array(
            [
                ((i / 255.0) ** (1.0 / gamma)) * 255
                for i in np.arange(256)
            ]
        ).astype("uint8")

        enhanced = cv2.LUT(
            enhanced,
            table
        )

        return enhanced, True, brightness


    # -----------------------------------------------------
    # LOW LIGHT
    # -----------------------------------------------------

    if brightness < LOW_LIGHT_THRESHOLD:

        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=1.6,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel
            )
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        return enhanced, True, brightness


    # -----------------------------------------------------
    # NORMAL LIGHT
    # -----------------------------------------------------

    return frame, False, brightness


# =========================================================
# FATIGUE SCORE
# =========================================================

def calculate_fatigue_score(
    eye_frames,
    yawn_frames,
    jaw_frames,
    distraction_confirmed
):

    score = 0.0

    if eye_frames > 0:

        score += min(
            eye_frames / EYE_FRAME_LIMIT,
            1.0
        ) * 45


    if yawn_frames > 0:

        score += min(
            yawn_frames / YAWN_FRAME_LIMIT,
            1.0
        ) * 25


    if jaw_frames > 0:

        score += min(
            jaw_frames / MASK_YAWN_FRAME_LIMIT,
            1.0
        ) * 15


    if distraction_confirmed:

        score += 15


    return min(
        score,
        100
    )


# =========================================================
# FATIGUE LEVEL
# =========================================================

def get_fatigue_level(score):

    if score < 30:
        return "LOW"

    elif score < 60:
        return "MODERATE"

    return "HIGH"


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(
    drowsiness,
    yawning,
    distraction,
    fatigue_level
):

    if drowsiness:
        return "HIGH"

    if distraction:
        return "HIGH"

    if fatigue_level == "HIGH":
        return "HIGH"

    if yawning:
        return "MEDIUM"

    if fatigue_level == "MODERATE":
        return "MEDIUM"

    return "LOW"


# =========================================================
# UI HELPERS
# =========================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.78
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (20, 25, 35),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (80, 90, 105),
        1
    )


def put_text(
    frame,
    text,
    position,
    scale=0.55,
    color=(235, 235, 235),
    thickness=1
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def status_color(status):

    if (
        "DROWSINESS" in status
        or "YAWNING" in status
        or "DISTRACTION" in status
    ):
        return (70, 70, 255)


    if status in [
        "LOOKING LEFT",
        "LOOKING RIGHT",
        "LOOKING UP",
        "LOOKING DOWN"
    ]:
        return (0, 190, 255)


    if status == "FACE NOT DETECTED":

        return (70, 70, 255)


    return (60, 220, 100)


# =========================================================
# HEAD INDICATOR
# =========================================================

def draw_head_indicator(
    frame,
    cx,
    cy,
    head_status
):

    radius = 48

    cv2.circle(
        frame,
        (cx, cy),
        radius,
        (220, 220, 220),
        2,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (cx - radius, cy),
        (cx + radius, cy),
        (90, 90, 100),
        1,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (cx, cy - radius),
        (cx, cy + radius),
        (90, 90, 100),
        1,
        cv2.LINE_AA
    )

    directions = {

        "LOOKING LEFT": (-30, 0),

        "LOOKING RIGHT": (30, 0),

        "LOOKING UP": (0, -30),

        "LOOKING DOWN": (0, 30),

        "ATTENTIVE": (0, 0)
    }

    dx, dy = directions.get(
        head_status,
        (0, 0)
    )


    if dx == 0 and dy == 0:

        cv2.circle(
            frame,
            (cx, cy),
            9,
            (80, 220, 120),
            -1,
            cv2.LINE_AA
        )

    else:

        cv2.arrowedLine(
            frame,
            (cx, cy),
            (cx + dx, cy + dy),
            status_color(head_status),
            4,
            cv2.LINE_AA,
            tipLength=0.35
        )


    label = "CENTER"

    if head_status != "ATTENTIVE":

        label = head_status.replace(
            "LOOKING ",
            ""
        )


    text_size = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1
    )[0]


    put_text(
        frame,
        label,
        (
            cx - text_size[0] // 2,
            cy + radius + 22
        ),
        0.48,
        status_color(head_status),
        1
    )


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    raise SystemExit


# Try HD
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)


print("=" * 65)
print(
    "AI DRIVER DROWSINESS & DISTRACTION DETECTION"
)
print(
    "Dashboard: http://127.0.0.1:5000"
)
print(
    "AI Model : "
    + ("LOADED" if ai_model_loaded else "NOT LOADED")
)
print(
    "Press ESC to stop."
)
print("=" * 65)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()


    if not success:

        print(
            "Could not read camera frame."
        )

        break


    # =====================================================
    # MIRROR / FLIP CAMERA
    # =====================================================
    # Show the webcam like a normal mirror/selfie view.
    frame = cv2.flip(frame, 1)


    # =====================================================
    # AUTO LIGHTING
    # =====================================================

    frame, low_light_mode, brightness = enhance_lighting(
        frame
    )


    h, w, _ = frame.shape


    # =====================================================
    # RGB
    # =====================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # FACE MESH
    # =====================================================

    results = face_mesh.process(
        rgb
    )


    # =====================================================
    # HEADER
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 58),
        (18, 22, 30),
        -1
    )


    put_text(
        frame,
        "AI DRIVER MONITOR",
        (24, 36),
        0.72,
        (245, 245, 245),
        2
    )


    put_text(
        frame,
        "REAL-TIME DROWSINESS • YAWN • DISTRACTION ANALYSIS",
        (285, 35),
        0.45,
        (155, 165, 180),
        1
    )


    # =====================================================
    # FACE DETECTED
    # =====================================================

    if results.multi_face_landmarks:

        # -------------------------------------------------
        # FACE FOUND
        # -------------------------------------------------

        face_lost_frames = 0

        last_valid_face_time = time.time()

        face_landmarks = (
            results.multi_face_landmarks[0]
        )


        # =================================================
        # EYES
        # =================================================

        left_eye_points = []

        right_eye_points = []


        for idx in LEFT_EYE:

            landmark = (
                face_landmarks.landmark[idx]
            )

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            left_eye_points.append(
                (x, y)
            )


        for idx in RIGHT_EYE:

            landmark = (
                face_landmarks.landmark[idx]
            )

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            right_eye_points.append(
                (x, y)
            )


        # =================================================
        # MOUTH
        # =================================================

        mouth_points = []


        for idx in MOUTH:

            landmark = (
                face_landmarks.landmark[idx]
            )

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            mouth_points.append(
                (x, y)
            )


        # =================================================
        # EAR
        # =================================================

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


        last_ear = ear


        # =================================================
        # MAR
        # =================================================

        mar = calculate_mar(
            mouth_points
        )


        last_mar = mar


        # =================================================
        # ADAPTIVE EAR CALIBRATION
        # =================================================

        if not ear_calibrated:

            elapsed_ear = time.time() - ear_calibration_start

            if elapsed_ear < EAR_CALIBRATION_TIME and ear > 0.15:
                ear_calibration_values.append(ear)

            elif elapsed_ear >= EAR_CALIBRATION_TIME:
                if len(ear_calibration_values) >= EAR_CALIBRATION_MIN_SAMPLES:
                    ear_baseline = float(np.median(ear_calibration_values))
                    adaptive_ear_threshold = max(
                        EAR_MIN_THRESHOLD,
                        min(EAR_MAX_THRESHOLD, ear_baseline * 0.72)
                    )
                else:
                    adaptive_ear_threshold = EAR_THRESHOLD

                ear_calibrated = True

        # =================================================
        # EYE DROWSINESS
        # =================================================

        if ear < adaptive_ear_threshold:

            eye_closed_frames += 1

        else:

            eye_closed_frames = 0


        # =================================================
        # NORMAL YAWN
        # =================================================

        if mar > MAR_THRESHOLD:

            mouth_open_frames += 1

        else:

            mouth_open_frames = 0


        # =================================================
        # JAW LANDMARKS
        # =================================================

        nose = (
            face_landmarks.landmark[1]
        )

        chin = (
            face_landmarks.landmark[152]
        )

        forehead = (
            face_landmarks.landmark[10]
        )


        nose_y = nose.y

        chin_y = chin.y

        forehead_y = forehead.y


        face_height = abs(
            chin_y -
            forehead_y
        )


        # =================================================
        # JAW POSITION
        # =================================================

        if face_height > 0.01:

            jaw_position = (
                chin_y -
                nose_y
            ) / face_height

        else:

            jaw_position = 0.0


        # =================================================
        # AI BEHAVIOR PREDICTION
        # =================================================
        # Additional AI layer; existing CV rules remain unchanged.

        raw_ai_prediction, raw_ai_confidence = predict_ai_behavior(
            ear,
            mar,
            jaw_position,
            brightness
        )

        ai_prediction, ai_confidence = smooth_ai_prediction(
            raw_ai_prediction,
            raw_ai_confidence
        )

        detection_data["ai_prediction"] = ai_prediction
        detection_data["ai_confidence"] = round(
            float(ai_confidence),
            1
        )

        # =================================================
        # JAW CALIBRATION
        # =================================================

        if jaw_baseline is None:

            elapsed_jaw = (
                time.time()
                - jaw_calibration_start
            )


            if elapsed_jaw < JAW_CALIBRATION_TIME:

                jaw_calibration_values.append(
                    jaw_position
                )

                put_text(
                    frame,
                    "INITIALIZING JAW SENSOR - KEEP MOUTH CLOSED",
                    (24, h - 25),
                    0.52,
                    (0, 210, 255),
                    1
                )


            else:

                if len(
                    jaw_calibration_values
                ) > 10:

                    jaw_baseline = float(
                        np.median(
                            jaw_calibration_values
                        )
                    )

                else:

                    jaw_baseline = jaw_position


        # =================================================
        # JAW DIFFERENCE
        # =================================================

        if jaw_baseline is not None:

            jaw_difference = (
                jaw_position -
                jaw_baseline
            )

        else:

            jaw_difference = 0.0


        # =================================================
        # JAW HISTORY
        # =================================================

        jaw_history.append(
            jaw_difference
        )


        if len(jaw_history) > MAX_JAW_HISTORY:

            jaw_history.pop(0)


        # =================================================
        # MASK YAWNING SUPPORT
        # =================================================

        # Small jaw motion should not be treated as a yawn.
        jaw_is_opening = (
            jaw_difference >
            JAW_MOVEMENT_THRESHOLD
        )

        if jaw_is_opening:
            jaw_yawn_frames += 1
        else:
            jaw_yawn_frames = 0


        # =================================================
        # DROWSINESS
        # =================================================

        drowsiness = (
            eye_closed_frames >=
            EYE_FRAME_LIMIT
        )


        # =================================================
        # YAWNING
        # =================================================

        normal_yawning = (
            mouth_open_frames >=
            YAWN_FRAME_LIMIT
        )


        jaw_excursion = 0.0
        if len(jaw_history) >= 8:
            jaw_excursion = max(jaw_history) - min(jaw_history)

        masked_yawning = (
            jaw_yawn_frames >= MASK_YAWN_FRAME_LIMIT
            and jaw_excursion >= 0.10
            and jaw_difference > JAW_MOVEMENT_THRESHOLD
        )


        cv_drowsiness = drowsiness
        cv_yawning = (
            normal_yawning
            or masked_yawning
        )

        # AI is used as a secondary confirmation layer.
        ai_drowsiness_confirmed = (
            ai_prediction == "DROWSY"
            and ai_confidence >= AI_HIGH_CONFIDENCE
        )

        # Do not allow the classifier alone to create a yawn from a
        # single mouth-open frame. Require stable AI + some mouth/jaw evidence.
        ai_yawning_confirmed = (
            ai_prediction == "YAWNING"
            and ai_confidence >= AI_YAWN_CONFIDENCE
            and (mar > 0.55 or jaw_difference > 0.06)
        )

        drowsiness = cv_drowsiness or ai_drowsiness_confirmed
        yawning = cv_yawning or ai_yawning_confirmed


        # =================================================
        # EVENT COUNTERS
        # =================================================

        if (
            yawning
            and not previous_yawning
        ):

            yawn_count += 1


        if (
            drowsiness
            and not previous_drowsiness
        ):

            drowsiness_count += 1


        previous_yawning = yawning

        previous_drowsiness = drowsiness


        # =================================================
        # HEAD POSE
        # =================================================

        image_points = []


        for idx in FACE_POINTS:

            landmark = (
                face_landmarks.landmark[idx]
            )

            image_points.append(
                (
                    int(landmark.x * w),
                    int(landmark.y * h)
                )
            )


        image_points = np.array(
            image_points,
            dtype=np.float64
        )


        focal_length = w

        center = (
            w / 2,
            h / 2
        )


        camera_matrix = np.array(
            [
                [
                    focal_length,
                    0,
                    center[0]
                ],

                [
                    0,
                    focal_length,
                    center[1]
                ],

                [
                    0,
                    0,
                    1
                ]
            ],
            dtype=np.float64
        )


        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64
        )


        model_points = np.array(
            [
                (0.0, 0.0, 0.0),

                (0.0, -330.0, -65.0),

                (-225.0, 170.0, -135.0),

                (225.0, 170.0, -135.0),

                (-150.0, -150.0, -125.0),

                (150.0, -150.0, -125.0)
            ],
            dtype=np.float64
        )


        success_pose, rotation_vector, translation_vector = cv2.solvePnP(

            model_points,

            image_points,

            camera_matrix,

            dist_coeffs,

            flags=cv2.SOLVEPNP_ITERATIVE
        )


        # =================================================
        # DEFAULT HEAD STATUS
        # =================================================

        head_status = "ATTENTIVE"

        relative_yaw = 0.0

        relative_pitch = 0.0

        roll = 0.0

        distraction_elapsed = 0.0


        # =================================================
        # HEAD ANGLES
        # =================================================

        if success_pose:

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )


            angles, _, _, _, _, _ = (
                cv2.RQDecomp3x3(
                    rotation_matrix
                )
            )


            pitch = float(
                angles[0]
            )

            yaw = float(
                angles[1]
            )

            roll = float(
                angles[2]
            )


            # =================================================
            # CALIBRATION
            # =================================================

            if not calibrated:

                calibration_yaw.append(
                    yaw
                )

                calibration_pitch.append(
                    pitch
                )


                elapsed = (
                    time.time()
                    - calibration_start_time
                )


                remaining = max(
                    0.0,
                    CALIBRATION_TIME -
                    elapsed
                )


                put_text(
                    frame,
                    "HEAD CALIBRATION",
                    (24, 95),
                    0.58,
                    (0, 210, 255),
                    2
                )


                put_text(
                    frame,
                    "Look straight at the camera",
                    (24, 120),
                    0.48,
                    (220, 220, 220),
                    1
                )


                put_text(
                    frame,
                    f"{remaining:.1f}s",
                    (24, 146),
                    0.55,
                    (0, 210, 255),
                    2
                )


                if elapsed >= CALIBRATION_TIME:

                    baseline_yaw = float(
                        np.median(
                            calibration_yaw
                        )
                    )

                    baseline_pitch = float(
                        np.median(
                            calibration_pitch
                        )
                    )

                    calibrated = True


            # =================================================
            # HEAD POSITION
            # =================================================

            if calibrated:

                relative_yaw = (
                    yaw -
                    baseline_yaw
                )


                relative_pitch = (
                    pitch -
                    baseline_pitch
                )


                # IMPORTANT:
                # Keep this convention stable.
                # It was already working correctly.

                if abs(
                    relative_yaw
                ) > YAW_THRESHOLD:

                    # Convert pose sign to the driver's own left/right perspective.
                    if relative_yaw > 0:

                        head_status = (
                            "LOOKING LEFT"
                        )

                    else:

                        head_status = (
                            "LOOKING RIGHT"
                        )


                elif abs(
                    relative_pitch
                ) > PITCH_THRESHOLD:

                    if relative_pitch > 0:

                        head_status = (
                            "LOOKING UP"
                        )

                    else:

                        head_status = (
                            "LOOKING DOWN"
                        )


                else:

                    head_status = (
                        "ATTENTIVE"
                    )


                # =================================================
                # DISTRACTION CONFIRMATION
                # =================================================

                if head_status != "ATTENTIVE":

                    if (
                        distraction_direction
                        != head_status
                    ):

                        distraction_direction = (
                            head_status
                        )

                        distraction_start_time = (
                            time.time()
                        )

                        distraction_confirmed = (
                            False
                        )


                    if (
                        distraction_start_time
                        is None
                    ):

                        distraction_start_time = (
                            time.time()
                        )


                    distraction_elapsed = (
                        time.time()
                        -
                        distraction_start_time
                    )


                    if (
                        distraction_elapsed
                        >= DISTRACTION_TIME
                    ):

                        distraction_confirmed = (
                            True
                        )


                else:

                    distraction_start_time = (
                        None
                    )

                    distraction_confirmed = (
                        False
                    )

                    distraction_direction = (
                        "NONE"
                    )

                    distraction_elapsed = 0.0


                # =================================================
                # DISTRACTION COUNTER
                # =================================================

                if (
                    distraction_confirmed
                    and not previous_distraction
                ):

                    distraction_count += 1


                previous_distraction = (
                    distraction_confirmed
                )


        # =====================================================
        # SAVE LAST HEAD STATE
        # =====================================================

        last_head_status = head_status

        last_relative_yaw = (
            relative_yaw
        )

        last_relative_pitch = (
            relative_pitch
        )

        last_roll = roll


        # =====================================================
        # FATIGUE
        # =====================================================

        fatigue_score = (
            calculate_fatigue_score(
                eye_closed_frames,
                mouth_open_frames,
                jaw_yawn_frames,
                distraction_confirmed
            )
        )


        fatigue_level = (
            get_fatigue_level(
                fatigue_score
            )
        )


        # =====================================================
        # RISK
        # =====================================================

        risk_level = (
            get_risk_level(
                drowsiness,
                yawning,
                distraction_confirmed,
                fatigue_level
            )
        )


        # =====================================================
        # FINAL STATUS
        # =====================================================

        if drowsiness:

            status_text = (
                "DROWSINESS DETECTED"
            )


        elif yawning:

            status_text = (
                "YAWNING DETECTED"
            )


        elif distraction_confirmed:

            status_text = (
                "DISTRACTION DETECTED"
            )


        elif head_status != "ATTENTIVE":

            status_text = head_status


        else:

            status_text = "ATTENTIVE"


        # =====================================================
        # SESSION TIME
        # =====================================================

        session_seconds = int(
            time.time()
            -
            session_start_time
        )


        minutes = (
            session_seconds // 60
        )

        seconds = (
            session_seconds % 60
        )


        session_time_text = (
            f"{minutes:02d}:{seconds:02d}"
        )


        # =====================================================
        # AUDIO ALERT
        # =====================================================

        alert_needed = (
            drowsiness
            or yawning
            or distraction_confirmed
        )


        if alert_needed:

            current_time = time.time()


            if (
                current_time
                -
                last_alarm_time
                >= ALARM_INTERVAL
            ):

                try:

                    winsound.Beep(
                        1500,
                        800
                    )

                except Exception:

                    pass


                last_alarm_time = (
                    current_time
                )


        # =====================================================
        # DASHBOARD DATA
        # =====================================================

        detection_data.update({

            "status": status_text,

            "risk_level": risk_level,

            "ear": float(ear),

            "mar": float(mar),
            
            "jaw_ratio": round(float(jaw_position), 3),

            "head_position": head_status,

            "drowsiness": bool(
                drowsiness
            ),

            "yawning": bool(
                yawning
            ),

            "distraction": bool(
                distraction_confirmed
            ),

            "lighting": (
                "LOW LIGHT - ENHANCED"
                if low_light_mode
                else "NORMAL"
            ),

            "brightness": round(
                brightness,
                1
            ),

            "face": True,

            "glasses": "SUPPORTED",

            "mask": "EAR + JAW SUPPORTED",

            "alert": bool(
                alert_needed
            ),

            "fatigue_score": round(
                float(fatigue_score),
                1
            ),

            "fatigue_level": (
                fatigue_level
            ),

            "yawn_count": (
                yawn_count
            ),

            "drowsiness_count": (
                drowsiness_count
            ),

            "distraction_count": (
                distraction_count
            ),

            "session_time": (
                session_time_text
            ),

            "ai_prediction": ai_prediction,
            "ai_confidence": round(
                float(ai_confidence),
                1
            ),
            "ai_model": (
                "LOADED"
                if ai_model_loaded
                else "NOT LOADED"
            )
        })


        # =====================================================
        # SUBTLE EYE / MOUTH LANDMARKS
        # =====================================================

        for x, y in (
            left_eye_points +
            right_eye_points
        ):

            cv2.circle(
                frame,
                (x, y),
                2,
                (80, 220, 120),
                -1
            )


        for x, y in mouth_points:

            cv2.circle(
                frame,
                (x, y),
                2,
                (90, 150, 255),
                -1
            )


        # =====================================================
        # FACE BOX
        # =====================================================

        all_face_points = []


        for landmark in (
            face_landmarks.landmark
        ):

            all_face_points.append(
                (
                    int(
                        landmark.x * w
                    ),
                    int(
                        landmark.y * h
                    )
                )
            )


        face_xs = [
            p[0]
            for p in all_face_points
        ]

        face_ys = [
            p[1]
            for p in all_face_points
        ]


        x_min = max(
            0,
            min(face_xs) - 12
        )

        y_min = max(
            58,
            min(face_ys) - 12
        )

        x_max = min(
            w,
            max(face_xs) + 12
        )

        y_max = min(
            h,
            max(face_ys) + 12
        )


        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            status_color(
                status_text
            ),
            2,
            cv2.LINE_AA
        )


        # =====================================================
        # LEFT METRICS PANEL
        # =====================================================

        draw_panel(
            frame,
            18,
            72,
            290,
            325
        )


        put_text(
            frame,
            "LIVE METRICS",
            (34, 100),
            0.58,
            (245, 245, 245),
            2
        )


        put_text(
            frame,
            f"EAR          {ear:.2f}",
            (34, 132),
            0.50,
            (80, 230, 130),
            1
        )


        put_text(
            frame,
            f"MAR          {mar:.2f}",
            (34, 158),
            0.50,
            (100, 160, 255),
            1
        )


        put_text(
            frame,
            f"EYE CLOSED   {eye_closed_frames}",
            (34, 187),
            0.46,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            f"MOUTH OPEN   {mouth_open_frames}",
            (34, 213),
            0.46,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            f"JAW ACTIVE   {jaw_yawn_frames}",
            (34, 239),
            0.46,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            f"BRIGHTNESS   {brightness:.0f}",
            (34, 265),
            0.46,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            (
                "AUTO ENHANCED"
                if low_light_mode
                else "NORMAL LIGHT"
            ),
            (34, 293),
            0.45,
            (
                (0, 210, 255)
                if low_light_mode
                else (80, 220, 130)
            ),
            1
        )


        # =====================================================
        # HEAD ORIENTATION PANEL
        # =====================================================

        panel_x1 = max(
            320,
            w // 2 - 165
        )

        panel_x2 = min(
            w - 320,
            w // 2 + 165
        )


        if panel_x2 > panel_x1:

            draw_panel(
                frame,
                panel_x1,
                72,
                panel_x2,
                235
            )


            put_text(
                frame,
                "HEAD ORIENTATION",
                (
                    panel_x1 + 18,
                    100
                ),
                0.55,
                (245, 245, 245),
                2
            )


            indicator_cx = (
                panel_x1 +
                panel_x2
            ) // 2


            indicator_cy = 145


            draw_head_indicator(
                frame,
                indicator_cx,
                indicator_cy,
                head_status
            )


            put_text(
                frame,
                f"Yaw {relative_yaw:+.1f}°",
                (
                    panel_x1 + 22,
                    207
                ),
                0.45,
                (210, 210, 220),
                1
            )


            put_text(
                frame,
                f"Pitch {relative_pitch:+.1f}°",
                (
                    panel_x1 + 105,
                    207
                ),
                0.45,
                (210, 210, 220),
                1
            )


            put_text(
                frame,
                f"Roll {roll:+.1f}°",
                (
                    panel_x1 + 205,
                    207
                ),
                0.45,
                (210, 210, 220),
                1
            )


        # =====================================================
        # RIGHT SYSTEM STATUS
        # =====================================================

        right_x1 = max(
            0,
            w - 305
        )


        draw_panel(
            frame,
            right_x1,
            72,
            w - 18,
            382
        )


        put_text(
            frame,
            "SYSTEM STATUS",
            (
                right_x1 + 16,
                100
            ),
            0.58,
            (245, 245, 245),
            2
        )


        put_text(
            frame,
            "FACE        DETECTED",
            (
                right_x1 + 16,
                130
            ),
            0.48,
            (80, 220, 130),
            1
        )


        put_text(
            frame,
            "EYES        TRACKING",
            (
                right_x1 + 16,
                157
            ),
            0.48,
            (80, 220, 130),
            1
        )


        put_text(
            frame,
            "MOUTH       TRACKING",
            (
                right_x1 + 16,
                184
            ),
            0.48,
            (80, 220, 130),
            1
        )


        put_text(
            frame,
            "GLASSES     SUPPORTED",
            (
                right_x1 + 16,
                211
            ),
            0.48,
            (220, 210, 100),
            1
        )


        put_text(
            frame,
            "MASK        EAR + JAW",
            (
                right_x1 + 16,
                238
            ),
            0.48,
            (220, 210, 100),
            1
        )


        put_text(
            frame,
            f"SESSION     {session_time_text}",
            (
                right_x1 + 16,
                265
            ),
            0.48,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            f"YAWNS       {yawn_count}",
            (
                right_x1 + 16,
                292
            ),
            0.48,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            f"DROWSINESS  {drowsiness_count}",
            (
                right_x1 + 16,
                319
            ),
            0.48,
            (220, 220, 220),
            1
        )


        put_text(
            frame,
            f"DISTRACTION {distraction_count}",
            (
                right_x1 + 16,
                346
            ),
            0.48,
            (220, 220, 220),
            1
        )

        ai_display = (
            ai_prediction
            if ai_prediction != "UNAVAILABLE"
            else "WAITING"
        )

        put_text(
            frame,
            f"AI MODEL    {ai_display}  {ai_confidence:.0f}%",
            (
                right_x1 + 16,
                373
            ),
            0.44,
            (170, 210, 255),
            1
        )

        put_text(
            frame,
            f"EAR LIMIT   {adaptive_ear_threshold:.3f}",
            (
                right_x1 + 16,
                398
            ),
            0.42,
            (180, 195, 210),
            1
        )


        # =====================================================
        # BOTTOM STATUS BAR
        # =====================================================

        bottom_y1 = h - 92


        draw_panel(
            frame,
            18,
            bottom_y1,
            w - 18,
            h - 18
        )


        current_color = status_color(
            status_text
        )


        put_text(
            frame,
            "CURRENT STATUS",
            (
                35,
                bottom_y1 + 28
            ),
            0.43,
            (155, 165, 180),
            1
        )


        put_text(
            frame,
            status_text,
            (
                35,
                bottom_y1 + 59
            ),
            0.72,
            current_color,
            2
        )


        risk_color = (
            (70, 70, 255)
            if risk_level == "HIGH"
            else
            (0, 190, 255)
            if risk_level == "MEDIUM"
            else
            (80, 220, 130)
        )


        put_text(
            frame,
            f"RISK: {risk_level}",
            (
                w - 250,
                bottom_y1 + 30
            ),
            0.52,
            risk_color,
            2
        )


        put_text(
            frame,
            (
                f"FATIGUE: "
                f"{fatigue_score:.0f}%"
                f"  |  "
                f"{fatigue_level}"
            ),
            (
                w - 330,
                bottom_y1 + 61
            ),
            0.45,
            (220, 220, 220),
            1
        )


        # =====================================================
        # DISTRACTION TIMER
        # =====================================================

        if (
            calibrated
            and head_status != "ATTENTIVE"
            and distraction_start_time
            is not None
        ):

            timer_value = min(
                time.time()
                -
                distraction_start_time,
                DISTRACTION_TIME
            )

        else:

            timer_value = 0.0


        put_text(
            frame,
            (
                f"Distraction timer: "
                f"{timer_value:.1f}/"
                f"{DISTRACTION_TIME:.1f}s"
            ),
            (
                320,
                h - 28
            ),
            0.42,
            (220, 200, 80),
            1
        )


    # =====================================================
    # FACE NOT DETECTED
    # =====================================================

    else:

        face_lost_frames += 1


        # -------------------------------------------------
        # TEMPORARY FACE LOSS
        # -------------------------------------------------

        if (
            face_lost_frames
            <= MAX_FACE_LOST_FRAMES
        ):

            # Do NOT immediately reset everything.
            # This prevents small MediaPipe tracking
            # glitches from causing visible jumps.

            put_text(
                frame,
                "TRACKING...",
                (
                    w // 2 - 70,
                    h // 2
                ),
                0.75,
                (0, 210, 255),
                2
            )


            put_text(
                frame,
                "Face temporarily lost - recovering",
                (
                    w // 2 - 175,
                    h // 2 + 32
                ),
                0.48,
                (220, 220, 220),
                1
            )


            detection_data.update({

                "status": "TRACKING...",

                "risk_level": "LOW",

                "ear": float(
                    last_ear
                ),

                "mar": float(
                    last_mar
                ),

                "head_position": (
                    last_head_status
                ),

                "drowsiness": False,

                "yawning": False,

                "distraction": False,

                "lighting": (
                    "LOW LIGHT - ENHANCED"
                    if low_light_mode
                    else "NORMAL"
                ),

                "brightness": round(
                    brightness,
                    1
                ),

                "face": True,

                "glasses": "SUPPORTED",

                "mask": "EAR + JAW SUPPORTED",

                "alert": False,

                "fatigue_score": round(
                    float(fatigue_score),
                    1
                ),

                "fatigue_level": (
                    fatigue_level
                ),

                "yawn_count": (
                    yawn_count
                ),

                "drowsiness_count": (
                    drowsiness_count
                ),

                "distraction_count": (
                    distraction_count
                ),

                "session_time": (
                    f"{int(time.time() - session_start_time) // 60:02d}:"
                    f"{int(time.time() - session_start_time) % 60:02d}"
                ),
                "ai_prediction": ai_prediction,
                "ai_confidence": round(
                    float(ai_confidence),
                    1
                ),
                "ai_model": (
                    "LOADED"
                    if ai_model_loaded
                    else "NOT LOADED"
                )
            })


        # -------------------------------------------------
        # REAL FACE LOSS
        # -------------------------------------------------

        else:

            eye_closed_frames = 0

            mouth_open_frames = 0

            jaw_yawn_frames = 0


            distraction_start_time = None

            distraction_confirmed = False

            distraction_direction = "NONE"

            previous_distraction = False

            ai_prediction = "UNAVAILABLE"
            ai_confidence = 0.0


            detection_data.update({

                "status": (
                    "FACE NOT DETECTED"
                ),

                "risk_level": "HIGH",

                "ear": 0.0,

                "mar": 0.0,

                "head_position": (
                    "FACE NOT DETECTED"
                ),

                "drowsiness": False,

                "yawning": False,

                "distraction": False,

                "lighting": (
                    "LOW LIGHT - ENHANCED"
                    if low_light_mode
                    else "NORMAL"
                ),

                "brightness": round(
                    brightness,
                    1
                ),

                "face": False,

                "alert": False,

                "fatigue_score": 0,

                "fatigue_level": "LOW",

                "yawn_count": (
                    yawn_count
                ),

                "drowsiness_count": (
                    drowsiness_count
                ),

                "distraction_count": (
                    distraction_count
                ),

                "session_time": (
                    f"{int(time.time() - session_start_time) // 60:02d}:"
                    f"{int(time.time() - session_start_time) % 60:02d}"
                ),
                "ai_prediction": "UNAVAILABLE",
                "ai_confidence": 0.0,
                "ai_model": (
                    "LOADED"
                    if ai_model_loaded
                    else "NOT LOADED"
                )
            })


            draw_panel(
                frame,
                18,
                90,
                w - 18,
                h - 100
            )


            put_text(
                frame,
                "FACE NOT DETECTED",
                (
                    w // 2 - 150,
                    h // 2
                ),
                0.9,
                (70, 70, 255),
                2
            )


            put_text(
                frame,
                "Please position your face inside the camera view",
                (
                    w // 2 - 235,
                    h // 2 + 35
                ),
                0.52,
                (220, 220, 220),
                1
            )


    # =====================================================
    # CAMERA WINDOW
    # =====================================================


    cv2.imshow(
        "AI Driver Monitoring System",
        frame
    )


    # =====================================================
    # ESC
    # =====================================================

    if (
        cv2.waitKey(1)
        & 0xFF
        == 27
    ):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()