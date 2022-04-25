import base64
import cv2
from keras.models import load_model
import numpy as np
import pafy
from collections import deque

import requests

IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 30
LRCN_model = load_model('model/model_mobile4.h5')
CLASSES_LIST = ['Explosion', 'Shoplifting', 'Normal', 'Fighting']

def predict_single_action(video_file_link, SEQUENCE_LENGTH =30):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    LRCN_model = load_model('model/model_mobile3.h5')
    CLASSES_LIST = ['Explosion', 'Shoplifting', 'Normal', 'Fighting']

    video = pafy.new(video_file_link)
    best = video.getbest(preftype="mp4")
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(best.url)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (224, 224))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    #preds
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    #pred_classes
    predicted_label = np.argsort(predicted_labels_probabilities)[-4:][::-1]

    # Get the class name using the retrieved index.
    #predicted_class_name = CLASSES_LIST[predicted_label]
    #classes
    predicted_class_name = [CLASSES_LIST[i] for i in predicted_label]
    #print(f"predicted_class_name: {predicted_class_name}")
    props = predicted_labels_probabilities[predicted_label]

    result = {}
    for c , p in zip(predicted_class_name, props):
        result[c] = round(p*100 ,2)


    # Display the predicted action along with the prediction confidence.
    #print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 
    video_reader.release()

    #return predicted_class_name, predicted_labels_probabilities[predicted_label]
    return result



def predict_on_video(SEQUENCE_LENGTH = 30):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    #video_reader = cv2.VideoCapture(video_file_path)
    "rtsp://admin:ACATNR@102.184.104.74:554/H.264"
    video_reader = cv2.VideoCapture(0)
    

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    "video_reader.get(cv2.CAP_PROP_FPS)"
    #video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                  #video_reader.get(cv2.CAP_PROP_FPS) , (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''
    jpg_as_text = ''
    flag = 'Normal'
    
    
    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            
            # Clear the Queue
            frames_queue.clear()
        
        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Send Frame of Anomaly action
        if predicted_class_name != flag and predicted_class_name != '':
            retval, buffer_img= cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer_img)
            data1 = {
                "value": jpg_as_text,
                "dt": "24/04/022",
                "zone": "4",
                "anomalyType": predicted_class_name,
                "anomalyPriority": "2"
            }
            flag = predicted_class_name
            print("Type:", flag)
            r = requests.post('https://datapostapi.conveyor.cloud/api/Values/',data = data1)
            print(r.text)
            #r.close()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    #video_writer.release()


predict_on_video()