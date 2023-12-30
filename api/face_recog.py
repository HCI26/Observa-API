import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions, distance as dst
from api.drawer import FaceDrawer, ImageDrawer
from models import UnknownVisitor,SavedVisitor,Visit
import uuid
from api import db
from collections import deque 


class FaceMatch:
    def __init__(self, is_known, identity, distance, x, y, w, h):
        self.is_known = is_known
        self.identity = identity
        self.distance = distance
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class FaceAnalysisResults:
    def __init__(self):
        self.emotions = []
        self.age = None
        self.gender = None

class Emotion:
    def __init__(self, name, score):
        self.name = name
        self.score = score

class VideoAnalysisSingleResult:
    def __init__(self):
        self.face_bounding_box = None
        self.recognition_matches = []
        self.emotions = []
        self.age = None
        self.gender = None


class VideoAnalysisOutput:
    def __init__(self, results, drawn_img):
        self.results = results
        self.drawn_img = drawn_img

class FaceRecognizer:
    def __init__(self):
        self.target_size = None
        self.model = None
        self.consecutive_frames = 0
        self.pending_unknown_visitor = None

    def load_model(self, model_name):
        self.target_size = functions.find_target_size(model_name=model_name)
        self.model = DeepFace.build_model(model_name=model_name)

    def find_in_db(
        self,
        img_path,
        current_user_id,
        saved_visitors,
        unknown_visitors,
        model_name="VGG-Face",
        distance_metric="cosine",
        enforce_detection=False,
        detector_backend="retinaface",
        align=True,
        normalization="base",
        silent=False
    ):
        """
        This function applies verification several times and find the identities in a database

        Parameters:
                img_path: exact image path, numpy array (BGR) or based64 encoded image.
                Source image can have many faces. Then, result will be the size of number of
                faces in the source image.

                db_path (string): You should store some image files in a folder and pass the
                exact folder path to this. A database image can also have many faces.
                Then, all detected faces in db side will be considered in the decision.

                model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
                Dlib, ArcFace, SFace or Ensemble

                distance_metric (string): cosine, euclidean, euclidean_l2

                enforce_detection (bool): The function throws exception if a face could not be detected.
                Set this to False if you don't want to get exception. This might be convenient for low
                resolution images.

                detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
                dlib, mediapipe or yolov8.

                align (boolean): alignment according to the eye positions.

                normalization (string): normalize the input image before feeding to model

                silent (boolean): disable some logging and progress bars

        Returns:
                List of list of FaceMatch, outer one is each face on the screen, inner list is the face possibilities for each face
        """

        tic = time.time()

        target_objs = functions.extract_faces(
            img=img_path,
            target_size=self.target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )

        results = []

        for target_img, target_region, _ in target_objs:
            if target_region["x"] > 600 or target_region["x"] < 60 or target_region["y"] > 400 or target_region["y"] < 40:
                continue
            target_embedding_obj = DeepFace.represent(
                img_path=target_img,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            target_representation = target_embedding_obj[0]["embedding"]
            if len(target_representation) == 0:
                continue

            closest_match = None
            closest_distance = float("inf")
            for saved_visitor in saved_visitors:
                source_representation = saved_visitor.embedding
                distance = dst.findCosineDistance(source_representation, target_representation)
                if distance < closest_distance and distance < 0.4:
                    closest_match = FaceMatch(
                        is_known=True,
                        identity=saved_visitor,
                        distance=distance,
                        x=target_region["x"],
                        y=target_region["y"],
                        w=target_region["w"],
                        h=target_region["h"]
                    )
                    closest_distance = distance

             # If no match found among known visitors, check unknown visitors
            unknown_visitor_found = False
            if not closest_match:
                for unknown_visitor in unknown_visitors:
                    distance = dst.findCosineDistance(unknown_visitor.embedding, target_representation)
                    if distance < closest_distance and distance < 0.4:  # Adjust threshold as needed
                        closest_match = FaceMatch(
                            is_known=False,
                            identity=unknown_visitor,  
                            distance=distance,
                            x=target_region["x"],
                            y=target_region["y"],
                            w=target_region["w"],
                            h=target_region["h"]
                        )
                        closest_distance = distance
                        unknown_visitor_found = True

            if not closest_match and not unknown_visitor_found:
                closest_match = FaceMatch(
                            is_known=False,
                            identity=None,
                            distance=None,
                            x=target_region["x"],
                            y=target_region["y"],
                            w=target_region["w"],
                            h=target_region["h"]
                        )
                # Check if this face matches the pending unknown visitor
                if self.pending_unknown_visitor and dst.findCosineDistance(
                    self.pending_unknown_visitor, target_representation) < 0.4:  # Adjust threshold as needed
                    self.consecutive_frames += 1
                    if self.consecutive_frames >= 50:
                        
                        x=target_region["x"]
                        y=target_region["y"]
                        w=target_region["w"]
                        h=target_region["h"]
                        face_img = img_path[y:y+h, x:x+w]  # Extract face region from the original image


                        image_filename = f"unknown_visitor_{uuid.uuid4()}.jpg"
                        image_path = os.path.join("unknown_visitors", image_filename)  
                        cv2.imwrite(image_path, face_img)
                        # Create the unknown visitor in the database
                        unknown_visitor = SavedVisitor(
                            user_id=current_user_id,
                            image_path=image_path,
                            embedding=self.pending_unknown_visitor
                        )
                        db.session.add(unknown_visitor)
                        db.session.commit()
                        
                        self.pending_unknown_visitor = None  # Reset for potential new unknown visitors
                else:
                    # Start tracking a new potential unknown visitor
                    self.consecutive_frames = 1
                    self.pending_unknown_visitor = target_representation

               

            results.append(closest_match)
        


        toc = time.time()

        if not silent:
            print("find function lasts ", toc - tic, " seconds")

        return results
    
class FaceAnalyzer:
     def __init__(self, enable_emotion=True, enable_age_gender=True):
        if enable_age_gender:
            self.age_gender_model = DeepFace.build_model(model_name="Age")
            self.gender_model = DeepFace.build_model(model_name="Gender")
        if enable_emotion:
            self.emotion_model = DeepFace.build_model(model_name="Emotion")
        self.enable_emotion = enable_emotion
        self.enable_age_gender = enable_age_gender

     def analyze(self, img_path, detector_backend="retinaface", enforce_detection=True):
        demographies = DeepFace.analyze(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            silent=True,
        )
        if len(demographies) > 0:
            demography = demographies[0]

            analysis_results = FaceAnalysisResults()

            if self.enable_emotion:
                emotions = demography["emotion"]
                analysis_results.emotions = [
                    Emotion(emotion_name, emotion_score)
                    for emotion_name, emotion_score in emotions.items()
                ]

            if self.enable_age_gender:
                apparent_age = demography["age"]
                dominant_gender = demography["dominant_gender"]
                gender = "M" if dominant_gender == "Man" else "W"
                analysis_results.age = apparent_age
                analysis_results.gender = gender

            return analysis_results

        return None

     

class VideoAnalyzer:
    def __init__(self, face_recognizer: FaceRecognizer, face_analyzer: FaceAnalyzer = None):
        self.face_recognizer = face_recognizer
        self.face_analyzer = face_analyzer
        self.cooldown_period = 5  # Seconds
        self.presence_duration = 10  # Seconds
        self.presence_threshold = 0.8  # Percentage of frames required for continuous presence
        self.last_seen_knownvisitors = {}
        self.last_seen_unknownvisitors = {}
        self.present_visitors_known = set()
        self.present_visitors_unknown = set()
        self.recent_presence_known = {}  # Track presence in recent frames
        self.recent_presence_unknown = {}  # Track presence in recent frames


    def track_visitor(self,result:FaceMatch,current_user_id):
        visitor_id = result.identity.id if result.identity else None
        last_seen_visitors = self.last_seen_knownvisitors 
        present_visitors = self.present_visitors_known
        recent_presence = self.recent_presence_known
        now = time.time()

         # Cooldown check
        # if visitor_id in last_seen_visitors and (now - last_seen_visitors[visitor_id]) < self.cooldown_period:
        #     return

        # Presence tracking
        if visitor_id not in present_visitors:
            # New visitor detected
            present_visitors.add(visitor_id)
            visit = Visit(
                visitor_id=visitor_id,
                user_id=current_user_id,
                timestamp=now
            )
            db.session.add(visit)
            db.session.commit()
            # recent_presence[visitor_id] = deque([True] * self.presence_duration, maxlen=self.presence_duration)
        # else:
            # recent_presence[visitor_id].append(True)  # Mark as present in this frame

        # Thresholding for continuous presence
        # if sum(recent_presence[visitor_id]) / len(recent_presence[visitor_id]) < self.presence_threshold:
        #     # Visitor has not been continuously present, create a new visit
        #     # if result.is_known:
        #     visit = Visit(
        #         visitor_id=visitor_id,
        #         user_id=current_user_id,
        #         timestamp=now
        #     )
        #     # else:
        #     #     visit = Visit(
        #     #         unknown_visitor_id=visitor_id,
        #     #         user_id=current_user_id,
        #     #         timestamp=now
        #     #     )
        #     db.session.add(visit)
        #     db.session.commit()

        last_seen_visitors[visitor_id] = now  # Update last seen timestamp

        # Remove visitors from "present" set if not detected for presence_duration
        # for visitor_id in list(present_visitors):
        #     if now - last_seen_visitors[visitor_id] > self.presence_duration:
        #         present_visitors.remove(visitor_id)

    def analyze_video(
        self,
        img,
        current_user_id,
        saved_visitors,
        unknown_visitors,
        model_name="VGG-Face",
        detector_backend="retinaface",
        distance_metric="cosine",
        enable_face_analysis=False,
        time_threshold=5,
        frame_threshold=1,
    ):
        pivot_img_size = 112  # face recognition result image

        raw_img = img.copy()

        image_drawer = ImageDrawer(img.copy())  # Use a copy of the image to avoid modifying the original
        face_drawer = FaceDrawer(image_drawer)

        freeze = True
        face_detected = False
        face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
        freezed_frame = 0
        tic = time.time()
        target_size = functions.find_target_size(model_name=model_name)
        results = []

        if freeze == False:
            # try:
            # just extract the regions to highlight in webcam
            face_objs = DeepFace.extract_faces(
                img_path=img,
                target_size=target_size,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            faces = []
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                face = (facial_area["x"],
                        facial_area["y"],
                        facial_area["w"],
                        facial_area["h"])
                face_drawer.draw_face_bounding_box(face)
                faces.append(face)

            
            # except:  # to avoid exception if no face detected
            #     faces = []

            if len(faces) == 0:
                face_detected = False
            else:
                face_detected = True

      

        if freeze == True:
            # Perform face recognition using the FaceRecognizer
            recognition_results = self.face_recognizer.find_in_db(
                img_path=raw_img,
                current_user_id = current_user_id,
                unknown_visitors=unknown_visitors,
                saved_visitors=saved_visitors,
                model_name=model_name,
                distance_metric=distance_metric,
                align=True,
            )

            
            for i, recognition_result in enumerate(recognition_results):
                result = VideoAnalysisSingleResult()
                result.face_bounding_box = (recognition_result.x, recognition_result.y, recognition_result.w, recognition_result.h)
                result.recognition_matches = recognition_result

                if enable_face_analysis:
                    analysis = self.face_analyzer.analyze(raw_img, detector_backend=detector_backend)
                    result.emotions = analysis.emotions if analysis else []
                    result.age = analysis.age
                    result.gender = analysis.gender

                results.append(result)

            for result in results:
                self.track_visitor(result.recognition_matches,current_user_id)
                face_drawer.draw_face_bounding_box(result.face_bounding_box)
                if result.recognition_matches.is_known:
                    face_drawer.draw_recognition_results(result.recognition_matches)
                if enable_face_analysis:
                    face_drawer.draw_analysis_results(result.emotions, result.age, result.gender)

        drawn_img = image_drawer.get_image()  # Get the modified image
                

        return VideoAnalysisOutput(results, drawn_img)

    
