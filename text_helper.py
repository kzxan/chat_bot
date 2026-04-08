from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

def put_kazakh_text(frame, text, position, font_size=30, color=(0, 255, 0)):
    """
    OpenCV frame-ге қазақша мәтін жазу
    """
    # OpenCV → PIL форматына өзгерту
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        # Windows үшін:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Linux үшін:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # PIL → OpenCV форматына қайтару
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)