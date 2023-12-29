import datetime
import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions, distance as dst
from sqlalchemy.orm import Query
from models import User,SavedVisitor
from api import db

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class FaceRecognizer:
    def __init__(self):
        self.target_size = None
        self.model = None

    def load_model(self, model_name):
        self.target_size = functions.find_target_size(model_name=model_name)
        self.model = DeepFace.build_model(model_name=model_name)

    def update_visitor_last_visit(self, face_obj: dict, visitor:SavedVisitor):
        """
        Updates the last_visited time for the identified visitor in the database.

        Args:
            session: SQLAlchemy session object.
            face_obj: Dictionary containing face information (identity, distance).
            user_id: User ID associated with the saved visitors.
        """
        if face_obj["distance"] <= 0.2:  # adjust threshold as needed
            identity = face_obj["identity"]
            if visitor:
                visitor.last_visited = datetime.datetime.utcnow()
                db.session.commit()
            
    def find_in_db(
        self,
        img_path,
        saved_visitors,
        model_name="VGG-Face",
        distance_metric="cosine",
        enforce_detection=True,
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
                This function returns list of pandas data frame. Each item of the list corresponding to
                an identity in the img_path.
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

        resp_obj = []

        for target_img, target_region, _ in target_objs:
            target_embedding_obj = DeepFace.represent(
                img_path=target_img,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            target_representation = target_embedding_obj[0]["embedding"]

            result_df = pd.DataFrame()
            result_df["x"] = target_region["x"]
            result_df["y"] = target_region["y"]
            result_df["w"] = target_region["w"]
            result_df["h"] = target_region["h"]

            distances = []
            identities = []
            visitors = []
            for saved_visitor in saved_visitors:
                source_representation = saved_visitor.embedding
                
                if distance_metric == "cosine":
                    distance = dst.findCosineDistance(source_representation, target_representation)
                elif distance_metric == "euclidean":
                    distance = dst.findEuclideanDistance(source_representation, target_representation)
                elif distance_metric == "euclidean_l2":
                    distance = dst.findEuclideanDistance(
                        dst.l2_normalize(source_representation),
                        dst.l2_normalize(target_representation),
                    )
                else:
                    raise ValueError(f"invalid distance metric passes - {distance_metric}")
                
                identities.append(saved_visitor.name)
                distances.append(distance)
                visitors.append(saved_visitor)

            result_df["distance"] = distances
            result_df["identity"] = identities
            result_df["visitor"] = visitors
            threshold = dst.findThreshold(model_name, distance_metric)
            # result_df = result_df.drop(columns=[f"{model_name}_representation"])
            result_df = result_df[result_df["distance"] <= 0.2]
            result_df = result_df.sort_values(
                by=["distance"], ascending=True
            ).reset_index(drop=True)
            if result_df["visitor"].shape[0] != 0:
                result_df["visitor"][0].last_visited = datetime.datetime.utcnow()
                db.session.commit()
            result_df = result_df.drop(columns=["visitor"])
            resp_obj.append(result_df)

        toc = time.time()

        if not silent:
            print("find function lasts ", toc - tic, " seconds")
        
        
        return resp_obj


class FaceAnalyzer:
    def __init__(self, enable_emotion=True, enable_age_gender=True):
        self.emotion_model = DeepFace.build_model(model_name="Emotion")
        self.age_gender_model = DeepFace.build_model(model_name="Age")
        self.gender_model = DeepFace.build_model(model_name="Gender")
        self.enable_emotion = enable_emotion
        self.enable_age_gender = enable_age_gender

    def analyze(self, img_path, detector_backend="retinaface", enforce_detection=False):
        demographies = DeepFace.analyze(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            silent=True,
        )

        if len(demographies) > 0:
            demography = demographies[0]

            if self.enable_emotion:
                emotion = demography["emotion"]
                emotion_df = pd.DataFrame(
                    emotion.items(), columns=["emotion", "score"]
                )
                emotion_df = emotion_df.sort_values(
                    by=["score"], ascending=False
                ).reset_index(drop=True)

            if self.enable_age_gender:
                apparent_age = demography["age"]
                dominant_gender = demography["dominant_gender"]
                gender = "M" if dominant_gender == "Man" else "W"
                analysis_report = str(int(apparent_age)) + " " + gender

            return emotion_df, analysis_report

        return None, None


class VideoAnalyzer:
    def __init__(self, face_recognizer: FaceRecognizer, face_analyzer: FaceAnalyzer = None):
        self.face_recognizer = face_recognizer
        self.face_analyzer = face_analyzer

    def analyze_video(
        self,
        img,
        saved_visitors,
        model_name="VGG-Face",
        detector_backend="retinaface",
        distance_metric="cosine",
        enable_face_analysis=False,
        time_threshold=5,
        frame_threshold=1,
    ):

        text_color = (255, 255, 255)
        pivot_img_size = 112  # face recognition result image

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        recognition_results = None
        freeze = False
        face_detected = False
        face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
        freezed_frame = 0
        tic = time.time()
        target_size = functions.find_target_size(model_name=model_name)
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
                faces.append(
                    (
                        facial_area["x"],
                        facial_area["y"],
                        facial_area["w"],
                        facial_area["h"],
                    )
                )
            # except:  # to avoid exception if no face detected
            #     faces = []

            if len(faces) == 0:
                face_detected = False
            else:
                face_detected = True

        if face_detected == True:
            face_included_frames += 1
            if face_included_frames >= frame_threshold:
                freeze = True
            else:
                freeze = False
        else:
            face_included_frames = 0
            freeze = False

        if freeze == True:
            # perform face recognition and facial analysis if no freezing
            tic = time.time()
            recognition_results = self.face_recognizer.find_in_db(
                img_path=raw_img,
                saved_visitors=saved_visitors,
                model_name=model_name,
                distance_metric=distance_metric,
                align=True,
            )

            if enable_face_analysis:
                emotion_df, analysis_report = self.face_analyzer.analyze(
                    img=raw_img, detector_backend=detector_backend
                )
            else:
                emotion_df, analysis_report = None, None

            toc = time.time()
            print("webcam analyze function lasts ", toc - tic, " seconds")

            # draw faces and labels on the webcam window
            for i, face in enumerate(faces):
                x, y, w, h = face

                # draw facial area
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                # draw face recognition result
                if len(recognition_results) > 0:
                    result_df = recognition_results[i]
                    if len(result_df) > 0:
                        closest_identity = result_df.loc[0, "identity"]
                        cv2.putText(
                            img,
                            closest_identity,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                        )

                # draw face analysis result
                if enable_face_analysis:
                    if emotion_df is not None:
                        dominant_emotion = emotion_df.loc[0, "emotion"]
                        cv2.putText(
                            img,
                            dominant_emotion,
                            (x + w + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                        )
                    if analysis_report is not None:
                        cv2.putText(
                            img,
                            analysis_report,
                            (x + w + 10, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                        )

        return img, recognition_results, freeze, face_detected


