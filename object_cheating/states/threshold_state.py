import reflex as rx

class ThresholdState(rx.State):
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.70
    duration_threshold: float = 5.0
    model1_confidence_threshold: float = 0.25
    model1_iou_threshold: float = 0.70
    model2_confidence_threshold: float = 0.25
    model2_iou_threshold: float = 0.70
    model3_confidence_threshold: float = 0.60
    model3_duration_threshold: float = 5.0
    model4_confidence_threshold: float = 0.30
    model4_iou_threshold: float = 0.45
    model4_action_confidence_threshold: float = 0.75
    model5_face_confidence_threshold: float = 0.50
    model5_emotion_confidence_threshold: float = 0.35
    model6_confidence_threshold: float = 0.10
    model6_iou_threshold: float = 0.45
    model7_confidence_threshold: float = 0.25
    model7_iou_threshold: float = 0.70
    cross_threshold_model: int = 1

    def prev_cross_threshold_model(self):
        if self.cross_threshold_model > 1:
            self.cross_threshold_model -= 1

    def next_cross_threshold_model(self):
        if self.cross_threshold_model < 7:
            self.cross_threshold_model += 1

    def increment_confidence(self):
        if self.confidence_threshold < 1.0:
            self.confidence_threshold += 0.01
            self.confidence_threshold = round(self.confidence_threshold, 2)

    def decrement_confidence(self):
        if self.confidence_threshold > 0.0:
            self.confidence_threshold -= 0.01
            self.confidence_threshold = round(self.confidence_threshold, 2)

    def increment_second_threshold(self, active_model: int):
        if active_model == 3:
            if self.duration_threshold < 10.0:
                self.duration_threshold += 0.1
                self.duration_threshold = round(self.duration_threshold, 1)
        else:
            if self.iou_threshold < 1.0:
                self.iou_threshold += 0.01
                self.iou_threshold = round(self.iou_threshold, 2)

    def decrement_second_threshold(self, active_model: int):
        if active_model == 3:
            if self.duration_threshold > 1.0:
                self.duration_threshold -= 0.1
                self.duration_threshold = round(self.duration_threshold, 1)
        else:
            if self.iou_threshold > 0.0:
                self.iou_threshold -= 0.01
                self.iou_threshold = round(self.iou_threshold, 2)

    def set_confidence_from_str(self, value: str):
        try:
            self.confidence_threshold = float(value)
        except ValueError:
            print("Invalid input for confidence threshold")

    def set_second_threshold_from_str(self, value: str, active_model: int):
        try:
            if active_model == 3:
                self.duration_threshold = float(value)
            else:
                self.iou_threshold = float(value)
        except ValueError:
            print("Invalid input for second threshold")

    def set_model_confidence_from_str(self, value: str, model_number: int):
        try:
            parsed_value = float(value)
            if model_number == 1:
                self.model1_confidence_threshold = parsed_value
            elif model_number == 2:
                self.model2_confidence_threshold = parsed_value
            elif model_number == 3:
                self.model3_confidence_threshold = parsed_value
            elif model_number == 4:
                self.model4_confidence_threshold = parsed_value
            elif model_number == 5:
                self.model5_face_confidence_threshold = parsed_value
            elif model_number == 6:
                self.model6_confidence_threshold = parsed_value
            elif model_number == 7:
                self.model7_confidence_threshold = parsed_value
        except ValueError:
            print("Invalid input for model confidence threshold")

    def set_model_second_threshold_from_str(self, value: str, model_number: int):
        try:
            parsed_value = float(value)
            if model_number == 1:
                self.model1_iou_threshold = parsed_value
            elif model_number == 2:
                self.model2_iou_threshold = parsed_value
            elif model_number == 3:
                self.model3_duration_threshold = parsed_value
            elif model_number == 4:
                self.model4_iou_threshold = parsed_value
            elif model_number == 5:
                self.model5_emotion_confidence_threshold = parsed_value
            elif model_number == 6:
                self.model6_iou_threshold = parsed_value
            elif model_number == 7:
                self.model7_iou_threshold = parsed_value
        except ValueError:
            print("Invalid input for model second threshold")

    def set_model4_action_confidence_from_str(self, value: str):
        try:
            self.model4_action_confidence_threshold = float(value)
        except ValueError:
            print("Invalid input for Model 4 action confidence threshold")

    def set_model_defaults(self, model_number: int):
        if model_number == 3:
            self.confidence_threshold = 0.6
            self.duration_threshold = 5.0
        elif model_number == 4:
            self.confidence_threshold = 0.30
            self.iou_threshold = 0.45
        elif model_number == 5:
            self.confidence_threshold = 0.50
            self.iou_threshold = 0.35
        elif model_number == 6:
            self.confidence_threshold = 0.10
            self.iou_threshold = 0.45
        elif model_number == 7:
            self.confidence_threshold = 0.25
            self.iou_threshold = 0.70
        else:
            self.confidence_threshold = 0.25
            self.iou_threshold = 0.70
