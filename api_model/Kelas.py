from flask import Flask, request
from flask_restful import Api, Resource
import cv2
from PIL import Image
import numpy as np
# import api_model.BaksaraConst as BaksaraConst
import BaksaraConst as BaksaraConst
from numpy import asarray


class Kelas(Resource):
    def __init__(self):
        self.class_names = BaksaraConst.CLASS_NAMES4
        self.final_model = BaksaraConst.MODELS
        self.bypass_class = BaksaraConst.TheBypass
        print(f"All Classses: {self.class_names}")

    def prep_predict_debug(self, image):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model.predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        max_index = np.argmax(pred)
        print(f"index argmax : {max_index}\n\
            pred : {pred}\n")
        prob = pred[0][max_index]
        names = self.class_names[max_index]
        return prob, names

    def fit_image(self, image=None, def_offset=10, canvas_size=128):
        # Edge detection using Canny edge detector
        # convert the image into grayscale
        imagez = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(imagez, 100, 200)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes on the original image
        # image_with_boxes = cv2.cvtColor(imagez, cv2.COLOR_GRAY2RGB)
        
        cx1 = []
        cy1 = []
        cx2 = []
        cy2 = []
        for contour in contours:
            # Get the bounding rectangle coordinates
            x, y, w, h = cv2.boundingRect(contour)
            cx1.append(x)
            cy1.append(y)
            cx2.append(x+w)
            cy2.append(y+h)
            # Draw a rectangle around the contour

        myx1 = min(cx1)
        myy1 = min(cy1)
        myx2 = max(cx2)
        myy2 = max(cy2)
        # cv2.rectangle(image_with_boxes, (myx1, myy1), (myx2, myy2), (255, 255, 0), 2)

        # Display the original image with bounding boxes
        # plt.figure(figsize=(15, 5))
        # plt.subplot(141)
        # plt.imshow(image_with_boxes)
        # plt.title('Original Image with Bounding Boxes')
        # plt.axis('on')

        # Extract the region of interest (ROI) from the input image based on the largest bounding box
        to_process = imagez[myy1:myy2, myx1:myx2]

        # Calculate the new size with aspect ratio preserved
        max_size = 128 - 2 * def_offset
        height, width = to_process.shape[:2]

        if height > width:
            new_height = max_size
            ratio = new_height / height
            new_width = int(width * ratio)
            offset_x = def_offset
            offset_y = int((128 - new_height) / 2)
        else:
            new_width = max_size
            ratio = new_width / width
            new_height = int(height * ratio)
            offset_x = int((128 - new_width) / 2)
            offset_y = def_offset

        # Resize the image with the calculated size
        resized_image = cv2.resize(to_process, (new_width, new_height))

        # Create the canvas with padding
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

        # calculate the x middle
        # 128 / 2 = 64
        x_start = 64-new_width//2
        y_start = 64-new_height//2
        canvas[y_start:y_start+new_height, x_start:x_start+new_width] = resized_image
        canvas = cv2.bitwise_not(canvas)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

        # Display the intermediate steps if requested
        # if show_steps:
        #     plt.subplot(142)
        #     plt.imshow(to_process, cmap='gray')
        #     plt.title('Edge-detected Image')
        #     plt.axis('on')

        #     plt.subplot(143)
        #     plt.imshow(canvas, cmap='gray')
        #     plt.title('Image after Resizing and Padding')
        #     plt.axis('on')

        #     plt.tight_layout()
        #     plt.show()

        return canvas

    def post(self):

        if 'image' not in request.files:
            response = {
                'error' : 'no image found'
            }
            return response
        file = request.files['image']
        class_input = request.form['actual_class']

        if class_input in self.bypass_class:
            response = {
                'class': class_input,
                'prob': '1.0'
            }
            return response
        #find actual_class index in 2D array
        # the array is in self.class_names
        model_class_idx = -1  # Inisialisasi dengan nilai default

        for i, sublist in enumerate(self.class_names):
            if class_input in sublist:
                model_class_idx = i
                break

        if model_class_idx == -1:
            response = {
                'error' : 'no image found class idx'
            }
            return response


        try:
            gray_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            # _, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)
            # ubah kode diatas menjadi adaptive threshold
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            image = self.fit_image(image = binary_image, def_offset=10, canvas_size=128)
            # baru ditambahkan

            predku, sorted_ranku = self.prep_predict(image, model_index=model_class_idx)
            response_class = class_input
            response_prob = self.rules(predku, sorted_ranku, class_input, model_index=model_class_idx)
            # print(f"[KELAS][SELESAI PROSES]: {response_class} {response_prob}")
            
            response = {
            'class': response_class,
            'prob': str(response_prob)
            }
            return response
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            # Handle the error condition appropriately
            response = {
            'error' : f"An error occurred: {str(e)}"
            }
            return response

    def PreprocessImageAsArray(self, image, show_output=False):
        im = cv2.resize(image, (128, 128))

        image_as_array = np.expand_dims(im, axis=0)
        scaled_image_as_array = np.true_divide(image_as_array, 255)

        return scaled_image_as_array

    def take_class(self, pred, sorted_ranks, class_input, model_index=0):
        inputted_class_rank = 1
        rank = 1
        for class_rank in sorted_ranks:
            if self.class_names[model_index][class_rank] == class_input:
                inputted_class_rank = class_rank
            rank += 1
        return class_input, pred[0][inputted_class_rank]

    def prep_predict(self, image, model_index=0):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model[model_index].predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        return pred, sorted_ranks

    def rules(self, pred, sorted_rank, class_input, model_index=0 ):
        res = []
        res.append(self.take_class(pred, sorted_rank, class_input, model_index=model_index))
        highest_tuple = max(res, key=lambda x: x[1])
        return highest_tuple[1]


