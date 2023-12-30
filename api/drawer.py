import cv2

class ImageDrawer:
    def __init__(self, img):
        self.img = img

    def draw_rectangle(self, x, y, w, h, color=(0, 255, 0), thickness=2):
        cv2.rectangle(self.img, (x, y), (x + w, y + h), color, thickness)

    def draw_text(self, text, x, y, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255, 255, 255), thickness=2):
        cv2.putText(self.img, text, (x, y), font_face, font_scale, color, thickness)

    def get_image(self):
        return self.img
    

class FaceDrawer:
    def __init__(self, image_drawer):
        self.image_drawer = image_drawer

    def draw_face_bounding_box(self, bounding_box):
        x, y, w, h = bounding_box
        self.image_drawer.draw_rectangle(x, y, w, h)

    def draw_recognition_results(self, match, text_offset_y=-10):
        # for match in matches:
        # identity = str(match.identity)  # Access identity information
        self.image_drawer.draw_text(match.identity.name, match.x, match.y + text_offset_y)

    def draw_analysis_results(self, emotions, age, gender, x_offset=10, y_offset=10):
        if emotions:
            dominant_emotion = emotions[0].name  # Access dominant emotion
            self.image_drawer.draw_text(dominant_emotion, x_offset, y_offset)
        if age and gender:
            analysis_report = f"{age} {gender}"  # Combine age and gender
            self.image_drawer.draw_text(analysis_report, x_offset, y_offset + 20)