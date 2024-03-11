from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
import numpy as np
from tensorflow import keras
from numpy import asarray

# self defined
import api_model.BaksaraConst as Baksara
# import BaksaraConst as Baksara

   
class Scanner(Resource):
    def __init__(self):
        self.MODEL = Baksara.MODEL
        self.CLASS_NAMES = Baksara.CLASS_NAMES
    # Dilation Function
    def begining_resize(self, image=None, h_desired=276):
        im_h = image.shape[0]
        im_w = image.shape[1]
        aspect_ratio = h_desired / im_h
        new_h = int(aspect_ratio * im_h)
        new_w = int(aspect_ratio * im_w)
        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h))
        return resized_image
    def dilate_image(self, image, kernel_size):
        # Set the kernel size and sigma
        sigma = 1.5
        # Define the kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform dilation
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        return dilated_image

    # Horizontal Projectile Profile Function
    def horizontal_pp(self, image_2, image, threshold):
        start_putih = []
        end_putih = []
        h_projection = np.sum(image, axis=1)
        # print(h_projection)
        for x in range(len(h_projection)):
            if h_projection[x] > threshold:
                if len(start_putih) == len(end_putih):
                    start_putih.append(x)
            else:
                if len(start_putih) == (len(end_putih)+1):
                    end_putih.append(x)
        if(len(start_putih) == len(end_putih)+1):
            end_putih.append(len(h_projection))

        h_segmentation = []
        for x in range(len(start_putih)):
            t = image_2[start_putih[x]:end_putih[x],:]
            h_segmentation.append(t)
        return h_segmentation

    # Vertical Projectile Profile Function
    def vertical_pp(self,image):
        # Step b: Buat histogram vertical
        v_projection = np.sum(image, axis=0)

        start_putih = []
        end_putih = []

        for x in range(len(v_projection)):
            if v_projection[x] > 0:
                if len(start_putih) == len(end_putih):
                    start_putih.append(x)
            else:
                if len(start_putih) == (len(end_putih)+1):
                    end_putih.append(x)
        if(len(start_putih) == len(end_putih)+1):
            end_putih.append(len(v_projection))
        
        v_segmentation = []
        for x in range(len(start_putih)):
            t = image[:, start_putih[x]:end_putih[x]]
            v_segmentation.append(t)

        return v_segmentation


    # Image to canvas function
    def image_to_canfas(self,image):
        # Read the image to be processed
        to_process = image

        # Calculate the new size with aspect ratio preserved
        max_size = 128 - 2 * 18
        height, width = to_process.shape[:2]

        if height > width:
            new_height = max_size
            ratio = new_height / height
            new_width = int(width * ratio)
            offset_x = 18
            offset_y = int((128 - new_height) / 2)
        else:
            new_width = max_size
            ratio = new_width / width
            new_height = int(height * ratio)
            offset_x = int((128 - new_width) / 2)
            offset_y = 18

        # Resize the image with the calculated size
        resized_image = cv2.resize(to_process, (new_width, new_height))

        # Create the canvas with padding
        canvas = np.zeros((128, 128), dtype=np.uint8)

        # calculate the x middle
        # 128 / 2 = 64
        x_start = 64-new_width//2
        y_start = 64-new_height//2
        canvas[y_start:y_start+new_height, x_start:x_start+new_width] = resized_image

        return canvas


    def segmentation(self, input_image):
        # Convert RGB into Grayscle
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        

        # Convert Grayscale into Binary Image
        # _, threshold_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
        # chenge threshold into adaptive gaussian threshold
        _, threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Negative transformation    
        binary_image = cv2.bitwise_not(threshold_image)
        
        

        # Perform Dilation
        dilated_image = self.dilate_image(threshold_image, 41)
        

        # FIRST HORIZONTAL PP
        h1_segmentation = self.horizontal_pp(threshold_image,dilated_image,  0)
        print(f"Horizontal 1 : {len(h1_segmentation)}\n")

        
        # mymultiplot(h1_segmentation, 1)
        # CHECK DONE SAMPAI SINI
        # Remove empty segments at the beginning and end (if any)
        h1_segmentation = [region for region in h1_segmentation if np.sum(region) > 0]
        # mymultiplot(h1_segmentation, 1)
        
        # FIRST VERTICAL PP
        h1_regions = len(h1_segmentation)
        v1_segmentation = []
        for image in h1_segmentation:
            image = self.begining_resize(image = image, h_desired = 276)
            temp_im = self.vertical_pp(image)
            temp_im = [region for region in temp_im if np.sum(region) > 0]
            v1_segmentation.append(temp_im)
        
        # SECOND HORIZONTAL PP
        v1_regions = len(v1_segmentation)
        h2_segmentation = []
        tebal_px = 19
        h2_treshold = int(255 * tebal_px)

        for i in range(v1_regions):
            num_images = len(v1_segmentation[i])
            h2_temp = []

            for j in range(num_images):
                h_segmentation = self.horizontal_pp(v1_segmentation[i][j], v1_segmentation[i][j], h2_treshold)
                # Remove empty segments at the beginning and end (if any)
                h_segmentation = [region for region in h_segmentation if np.sum(region) > 0]

                h2_temp.append(h_segmentation)
            h2_segmentation.append(h2_temp)

        # SECOND VERTICAL PP
        h2_regions = len(h2_segmentation)
        v2_segmentation = []

        for i in range(h2_regions):
            num_col = len(h2_segmentation[i])
            # Menyimpan per baris
            v2_temp = []
            
            for j in range(num_col):
                num_img = len(h2_segmentation[i][j])
                if num_img == 1:
                    v2_temp.append(h2_segmentation[i][j])
                
                else:
                    # Menyimpan per satu set aksara dalam satu baris yang sama
                    v2_temp_2 = []
                    for k in range(num_img):
                        v_segmentation = self.vertical_pp(h2_segmentation[i][j][k])
                        # Remove empty segments at the beginning and end (if any)
                        v_segmentation = [region for region in v_segmentation if np.sum(region) > 0]

                        if len(v_segmentation) > 1:
                            v2_temp.append([v_segmentation[0]])
                            v2_temp_2.append(v_segmentation[1])
                        else:
                            v2_temp_2.append(v_segmentation[0])
                    
                    if (len(v2_temp_2) != 0):
                        v2_temp.append(v2_temp_2)
            v2_segmentation.append(v2_temp)

        # Set each image in the center of 128x128 canvas and perform bitwise not
        v2_regions = len(v2_segmentation)
        for i in range(v2_regions):
            num_col = len(v2_segmentation[i])
            for j in range(num_col):
                num_img = len(v2_segmentation[i][j])
                for k in range(num_img):
                    v2_segmentation[i][j][k] = cv2.bitwise_not(self.image_to_canfas(v2_segmentation[i][j][k]))

        return v2_segmentation


    def PreprocessInput(self, image):
        # Resize the image to match the model's input shape
        image = cv2.resize(image, (128, 128)) 
        # Add batch dimension
        image_as_array = np.expand_dims(asarray(image), axis=0)
        scaled_image_as_array = np.true_divide(image_as_array, 255)
        return scaled_image_as_array

    def classification(self, array_images):
        # Load Model
        
        final_model  = self.MODEL

        # Define Class Names
        class_names = self.CLASS_NAMES
        # Preprocess Function
        

        result = []

        # Display the number of segmented regions
        regions = len(array_images)
        for i in range(regions):
            num_col = len(array_images[i])
            result_col = []
            num_images = 0    
            for j in range(num_col):
                num_img = len(array_images[i][j])
                num_images += num_img

            if num_images == 1:
                input_image = array_images[i][0][0]
                # Convert Grayscale to RGB
                if len(input_image.shape) == 2:  
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                input_image = self.PreprocessInput(input_image)
                # Prediction
                pred = final_model.predict(input_image)
                sorted_ranks = np.flip(np.argsort(pred[0]))
                result_col.append(class_names[sorted_ranks[0]])
            else:
                for j in range(num_col):
                    result_imgs = []
                    num_img = len(array_images[i][j])

                    for k in range(num_img):
                        input_image = array_images[i][j][k]
                        # Convert Grayscale to RGB
                        if len(input_image.shape) == 2: 
                            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                        input_image = self.PreprocessInput(input_image)  
                        # Prediction
                        pred = final_model.predict(input_image)
                        sorted_ranks = np.flip(np.argsort(pred[0]))
                        result_imgs.append(class_names[sorted_ranks[0]])
                        # print(result_imgs)
                    if (len(result_imgs) > 0):
                        result_col.append(result_imgs)

            result.append(result_col)
        return result

    def transliteration(self, pred_result):
        final_result = []
        num_row = len(pred_result)
        for i in range(num_row):
            row_result = []
            probably_o = False
            temp_o = []
            num_col = len(pred_result[i])
            for j in range(num_col):
                num_img = len(pred_result[i][j])
                for k in range(num_img):
                    # print(f"pred result\t:{pred_result[i][j][k].split('_')}")
                    jenis, aksara = pred_result[i][j][k].split('_')
                    if jenis == 'carakan':
                        if probably_o:
                            if len(temp_o) == 1:
                                temp_o.append(aksara)
                            elif len(temp_o) == 2:
                                # Aksara using sandangan e not o
                                row_result.append(temp_o[1][:-1] + temp_o[0])
                                probably_o = False
                                temp_o = []
                                row_result.append(aksara)
                            elif len(temp_o) == 3 and temp_o[-1] in ['h', 'ng', 'r']:
                                # sandhangan è using 'h', 'ng', 'r'
                                row_result.append(temp_o[1][:-1] + temp_o[0] + temp_o[2])
                                probably_o = False
                                temp_o = []
                                row_result.append(aksara)
                            else:
                                row_result.append(aksara)
                        else:
                            row_result.append(aksara)
                    elif jenis == 'sandhangan':
                        # taling 'è'
                        if aksara == 'e2':
                            probably_o = True
                            temp_o.append('è')
                        # tarung 'o'
                        elif aksara == 'o':
                            if probably_o and len(temp_o) == 2:
                                row_result.append(temp_o[1][:-1] + aksara)
                                probably_o = False
                                temp_o = []
                            # Terdeteksi o namun tidak ada taling, kemungkinan missclassify h
                            else:
                                row_result.append('h')
                        # e, i, u
                        elif aksara in ['e', 'i', 'u']:
                            if row_result:
                                row_result[-1] = row_result[-1][:-1] + aksara
                        # h, ng, r
                        else:
                            if probably_o and len(temp_o) == 2:
                                temp_o.append(aksara)
                            elif row_result:
                                row_result[-1] = row_result[-1] + aksara
            final_result.append(row_result)
        return final_result

    def fit_image(self, imagez = None, def_offset = 10 ):
        contours, _ = cv2.findContours(imagez, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            # cv2.rectangle(image_with_rectangles, (x, y), (x+w, y+h), (255, 255, 0), 2)

        myx1 = min(cx1)
        myy1 = min(cy1)
        myx2 = max(cx2)
        myy2 = max(cy2)
        # cv2.rectangle(image_with_rectangles, (myx1, myy1), (myx2, myy2), (0, 255, 0), 2)
        # Read the image to be processed
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
        canvas = np.zeros((128, 128), dtype=np.uint8)

        # calculate the x middle
        # 128 / 2 = 64
        x_start = 64-new_width//2
        y_start = 64-new_height//2
        canvas[y_start:y_start+new_height, x_start:x_start+new_width] = resized_image
        canvas = cv2.bitwise_not(canvas)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return canvas
    def post(self, image = None):
        if 'image' not in request.files:
            response = {
                'error' : 'no image found'
            }
            return response
        file = request.files['image']
        
        image = None
        try:
            gray_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            
            bottom = 80
            top = 255

            _, binary_image = cv2.threshold(gray_image, bottom, top, cv2.THRESH_BINARY)
            image = binary_image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            response = { "error": f"Image undetected :: {str(e)}" }
            return response

        segmentation_result = None
        try:
            segmentation_result = self.segmentation(image)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            response = { "error": "image segmentation fault" }
            return response
        
        classification_result = None
        try:
            classification_result = self.classification(segmentation_result)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            response = { "error": "classification fault" }
            return response
        
        transliteration_result = None
        try:
            transliteration_result = self.transliteration(classification_result)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            response = { "error": "transliteration fault" }
            return response
        

        print(transliteration_result)
        response = {
            "result" : transliteration_result
        }
        return response
