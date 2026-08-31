import math
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pygame


# ============================================================
# CONFIG
# ============================================================

WIDTH = 800
HEIGHT = 600

CAMERA_INDEX = 0

# 300 partikel cukup ringan
NUM_PARTICLES = 300

# MediaPipe diproses setiap 2 frame
PROCESS_EVERY = 2

MODEL_PATH = Path("hand_landmarker.task")


# ============================================================
# PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Hand Gesture Particle"
)

clock = pygame.time.Clock()


# Font UI
ui_font = pygame.font.Font(
    None,
    24
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("ERROR: hand_landmarker.task tidak ditemukan.")

    pygame.quit()
    raise SystemExit


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


options = mp.tasks.vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.6,

    min_hand_presence_confidence=0.6,

    min_tracking_confidence=0.6
)


hand_landmarker = (
    mp.tasks.vision.HandLandmarker
    .create_from_options(options)
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    print("ERROR: Kamera tidak dapat dibuka.")

    hand_landmarker.close()
    pygame.quit()

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)


# ============================================================
# PARTICLE
# ============================================================

class Particle:

    def __init__(self):

        # Posisi random
        self.x = random.uniform(
            0,
            WIDTH
        )

        self.y = random.uniform(
            0,
            HEIGHT
        )

        # Kecepatan idle
        self.vx = random.uniform(
            -1.0,
            1.0
        )

        self.vy = random.uniform(
            -1.0,
            1.0
        )

        # Ukuran
        self.radius = random.choice(
            [1, 1, 1, 2]
        )

        # Warna awal
        self.color = (
            190,
            210,
            240
        )


    # --------------------------------------------------------
    # IDLE
    # --------------------------------------------------------

    def idle(self):

        self.x += self.vx
        self.y += self.vy


        if self.x < 0:

            self.x = 0
            self.vx *= -1


        if self.x > WIDTH:

            self.x = WIDTH
            self.vx *= -1


        if self.y < 0:

            self.y = 0
            self.vy *= -1


        if self.y > HEIGHT:

            self.y = HEIGHT
            self.vy *= -1


    # --------------------------------------------------------
    # MOVE TO TARGET
    # --------------------------------------------------------

    def move_to(
        self,
        target_x,
        target_y,
        speed
    ):

        self.x += (
            target_x - self.x
        ) * speed

        self.y += (
            target_y - self.y
        ) * speed


    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    def draw(self):

        pygame.draw.circle(

            screen,

            self.color,

            (
                int(self.x),
                int(self.y)
            ),

            self.radius
        )


# ============================================================
# HEART
# ============================================================

def heart_point(t):

    x = 16 * (
        math.sin(t) ** 3
    )

    y = -(
        13 * math.cos(t)
        - 5 * math.cos(2 * t)
        - 2 * math.cos(3 * t)
        - math.cos(4 * t)
    )

    return (

        WIDTH / 2 + x * 11,

        HEIGHT / 2 + y * 11

    )


# ============================================================
# STAR
# ============================================================

def star_point(t):

    radius = 145 * (
        1 + 0.5 * math.sin(5 * t)
    )

    return (

        WIDTH / 2
        + radius * math.cos(t),

        HEIGHT / 2
        + radius * math.sin(t)

    )


# ============================================================
# CIRCLE
# ============================================================

def circle_point(t):

    radius = 210

    return (

        WIDTH / 2
        + radius * math.cos(t),

        HEIGHT / 2
        + radius * math.sin(t)

    )


# ============================================================
# SPIRAL
# ============================================================

def spiral_point(t):

    radius = 15 + 150 * (
        t / (2 * math.pi)
    )

    angle = t * 5

    return (

        WIDTH / 2
        + radius * math.cos(angle),

        HEIGHT / 2
        + radius * math.sin(angle)

    )


# ============================================================
# CREATE LOVE TARGET
# ============================================================

def create_love_targets():

    """
    Membuat titik target berdasarkan
    tulisan LOVE.

    Titik teks dibuat lebih banyak dahulu,
    kemudian dipilih secara merata.
    """

    # --------------------------------------------------------
    # FONT WINDOWS
    # --------------------------------------------------------

    font_path = r"C:\Windows\Fonts\arialbd.ttf"


    # Jika Arial Bold tersedia
    if Path(font_path).exists():

        love_font = pygame.font.Font(
            font_path,
            170
        )

    else:

        # Fallback
        love_font = pygame.font.Font(
            None,
            170
        )


    # --------------------------------------------------------
    # RENDER TEXT
    # --------------------------------------------------------

    text_surface = love_font.render(

        "LOVE",

        True,

        (255, 255, 255)

    )


    text_width = text_surface.get_width()

    text_height = text_surface.get_height()


    # --------------------------------------------------------
    # CENTER TEXT
    # --------------------------------------------------------

    offset_x = (
        WIDTH - text_width
    ) // 2

    offset_y = (
        HEIGHT - text_height
    ) // 2


    # --------------------------------------------------------
    # COLLECT ALL TEXT PIXELS
    # --------------------------------------------------------

    all_points = []


    # Sampling setiap 3 pixel
    # agar proses pembuatan target ringan
    step = 3


    for y in range(
        0,
        text_height,
        step
    ):

        for x in range(
            0,
            text_width,
            step
        ):

            # Alpha pixel
            alpha = text_surface.get_at(
                (x, y)
            ).a


            if alpha > 0:

                all_points.append(

                    (
                        x + offset_x,
                        y + offset_y
                    )

                )


    # --------------------------------------------------------
    # RANDOMIZE
    # --------------------------------------------------------

    random.shuffle(
        all_points
    )


    # --------------------------------------------------------
    # PILIH PARTIKEL SECARA MERATA
    # --------------------------------------------------------

    if len(all_points) <= NUM_PARTICLES:

        targets = all_points

    else:

        targets = []

        total = len(all_points)


        for i in range(
            NUM_PARTICLES
        ):

            index = int(

                i
                * total
                / NUM_PARTICLES

            )

            targets.append(
                all_points[index]
            )


    # --------------------------------------------------------
    # JIKA TARGET KURANG
    # --------------------------------------------------------

    while len(targets) < NUM_PARTICLES:

        targets.append(

            random.choice(
                all_points
            )

        )


    return targets


# ============================================================
# PARTICLES
# ============================================================

particles = [

    Particle()

    for _ in range(
        NUM_PARTICLES
    )

]


# ============================================================
# SHAPE TARGETS
# ============================================================

heart_targets = [

    heart_point(
        2 * math.pi * i / NUM_PARTICLES
    )

    for i in range(
        NUM_PARTICLES
    )

]


star_targets = [

    star_point(
        2 * math.pi * i / NUM_PARTICLES
    )

    for i in range(
        NUM_PARTICLES
    )

]


circle_targets = [

    circle_point(
        2 * math.pi * i / NUM_PARTICLES
    )

    for i in range(
        NUM_PARTICLES
    )

]


spiral_targets = [

    spiral_point(
        2 * math.pi * i / NUM_PARTICLES
    )

    for i in range(
        NUM_PARTICLES
    )

]


# ============================================================
# LOVE
# ============================================================

love_targets = create_love_targets()


# ============================================================
# FINGER DETECTION
# ============================================================

def finger_up(
    landmarks,
    tip,
    pip
):

    return (
        landmarks[tip].y
        <
        landmarks[pip].y
    )


# ============================================================
# THUMB
# ============================================================

def thumb_open(
    landmarks
):

    distance = math.hypot(

        landmarks[4].x
        - landmarks[5].x,

        landmarks[4].y
        - landmarks[5].y

    )

    return distance > 0.12


# ============================================================
# GESTURE
# ============================================================

def detect_gesture(
    landmarks
):

    index = finger_up(
        landmarks,
        8,
        6
    )

    middle = finger_up(
        landmarks,
        12,
        10
    )

    ring = finger_up(
        landmarks,
        16,
        14
    )

    pinky = finger_up(
        landmarks,
        20,
        18
    )

    thumb = thumb_open(
        landmarks
    )


    # --------------------------------------------------------
    # PINCH
    # --------------------------------------------------------

    pinch_distance = math.hypot(

        landmarks[4].x
        - landmarks[8].x,

        landmarks[4].y
        - landmarks[8].y

    )


    if pinch_distance < 0.055:

        return "PINCH"


    # --------------------------------------------------------
    # FIVE
    # --------------------------------------------------------

    if (

        thumb
        and index
        and middle
        and ring
        and pinky

    ):

        return "FIVE"


    # --------------------------------------------------------
    # TWO
    # --------------------------------------------------------

    if (

        index
        and middle
        and not ring
        and not pinky

    ):

        return "TWO"


    # --------------------------------------------------------
    # ONE
    # --------------------------------------------------------

    if (

        index
        and not middle
        and not ring
        and not pinky

    ):

        return "ONE"


    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------

    if (

        not index
        and not middle
        and not ring
        and not pinky

    ):

        return "FIST"


    return "IDLE"


# ============================================================
# STATE
# ============================================================

gesture_state = "IDLE"

frame_counter = 0

timestamp_ms = 0

running = True


# ============================================================
# MAIN LOOP
# ============================================================

while running:


    # ========================================================
    # EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


    # ========================================================
    # CAMERA
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        break


    # Mirror
    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    frame_counter += 1


    if frame_counter % PROCESS_EVERY == 0:


        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb

        )


        timestamp_ms = int(

            time.monotonic()
            * 1000

        )


        results = (
            hand_landmarker
            .detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        if results.hand_landmarks:

            landmarks = (
                results.hand_landmarks[0]
            )


            gesture_state = (
                detect_gesture(
                    landmarks
                )
            )

        else:

            gesture_state = "IDLE"


    # ========================================================
    # BACKGROUND
    # ========================================================

    screen.fill(
        (7, 7, 17)
    )


    # ========================================================
    # FIST -> HEART
    # ========================================================

    if gesture_state == "FIST":


        for i, particle in enumerate(
            particles
        ):

            particle.color = (
                255,
                40,
                90
            )


            tx, ty = heart_targets[i]


            particle.move_to(

                tx,
                ty,

                0.06

            )


            particle.draw()


    # ========================================================
    # ONE -> LOVE
    # ========================================================

    elif gesture_state == "ONE":


        # ====================================================
        # HEARTBEAT
        # ====================================================

        current_time = time.monotonic()


        # Gelombang heartbeat
        beat = math.sin(
            current_time * 7
        )


        # Besar-kecil LOVE
        pulse = (
            1.0
            + beat * 0.045
        )


        # ====================================================
        # PARTICLES -> LOVE
        # ====================================================

        for i, particle in enumerate(
            particles
        ):


            particle.color = (
                255,
                45,
                120
            )


            base_x, base_y = (
                love_targets[i]
            )


            # ------------------------------------------------
            # SCALE DARI TENGAH
            # ------------------------------------------------

            tx = (

                WIDTH / 2

                + (

                    base_x
                    - WIDTH / 2

                ) * pulse

            )


            ty = (

                HEIGHT / 2

                + (

                    base_y
                    - HEIGHT / 2

                ) * pulse

            )


            # ------------------------------------------------
            # BERKUMPUL PERLAHAN
            # ------------------------------------------------

            particle.move_to(

                tx,
                ty,

                0.045

            )


            particle.draw()


        # ====================================================
        # HEARTBEAT INDICATOR
        # ====================================================

        beat_radius = int(

            7
            + beat * 2

        )


        pygame.draw.circle(

            screen,

            (255, 60, 130),

            (
                WIDTH // 2,
                65
            ),

            max(
                3,
                beat_radius
            )

        )


    # ========================================================
    # TWO -> STAR
    # ========================================================

    elif gesture_state == "TWO":


        for i, particle in enumerate(
            particles
        ):

            particle.color = (
                255,
                215,
                40
            )


            tx, ty = star_targets[i]


            particle.move_to(

                tx,
                ty,

                0.06

            )


            particle.draw()


    # ========================================================
    # FIVE -> CIRCLE
    # ========================================================

    elif gesture_state == "FIVE":


        for i, particle in enumerate(
            particles
        ):

            particle.color = (
                70,
                230,
                160
            )


            tx, ty = circle_targets[i]


            particle.move_to(

                tx,
                ty,

                0.06

            )


            particle.draw()


    # ========================================================
    # PINCH -> SPIRAL
    # ========================================================

    elif gesture_state == "PINCH":


        for i, particle in enumerate(
            particles
        ):

            particle.color = (
                210,
                90,
                250
            )


            tx, ty = spiral_targets[i]


            particle.move_to(

                tx,
                ty,

                0.055

            )


            particle.draw()


    # ========================================================
    # IDLE
    # ========================================================

    else:


        for particle in particles:

            particle.color = (
                190,
                210,
                240
            )


            particle.idle()

            particle.draw()


    # ========================================================
    # WEBCAM PREVIEW
    # ========================================================

    preview = cv2.resize(

        frame,

        (160, 120)

    )


    preview = cv2.cvtColor(

        preview,

        cv2.COLOR_BGR2RGB

    )


    # Pygame orientation
    preview = preview.swapaxes(
        0,
        1
    )


    surface = pygame.surfarray.make_surface(
        preview
    )


    screen.blit(

        surface,

        (20, HEIGHT - 140)

    )


    # ========================================================
    # STATUS
    # ========================================================

    status = {

        "IDLE":
            "IDLE",

        "FIST":
            "FIST -> HEART",

        "ONE":
            "1 FINGER -> LOVE",

        "TWO":
            "2 FINGERS -> STAR",

        "FIVE":
            "5 FINGERS -> CIRCLE",

        "PINCH":
            "PINCH -> SPIRAL"

    }.get(

        gesture_state,

        "IDLE"

    )


    status_surface = ui_font.render(

        status,

        True,

        (255, 255, 255)

    )


    screen.blit(

        status_surface,

        (20, 20)

    )


    # ========================================================
    # LOVE TEXT
    # ========================================================

    if gesture_state == "ONE":


        love_status = ui_font.render(

            "SHOWING LOVE...",

            True,

            (255, 80, 140)

        )


        screen.blit(

            love_status,

            (20, 50)

        )


    # ========================================================
    # DISPLAY
    # ========================================================

    pygame.display.flip()


    # FPS
    clock.tick(60)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hand_landmarker.close()

pygame.quit()