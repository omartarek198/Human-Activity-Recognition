
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for videos - for video processing.
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)
# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

def VideoPathToFrames(path,display=False):
    # input: path to video
    # output: list of frames in the video
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        return None
    frames = []
    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            if display:
                cv2.imshow('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break




        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return frames


def detectPose(image_pose, targetjoints=[], draw=False, display=False):
    if len(targetjoints) == 0:
        targetjoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    else:
        targetjoints = sorted(targetjoints)

    original_image = image_pose.copy()

    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)

    resultant = pose.process(image_in_RGB)
    if not resultant.pose_landmarks:
        return None
    if resultant.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),

                                                                                 thickness=2, circle_radius=2))

    # two pointers approach to find intersection between lists in o(n+m) instead of o(n*m)

    extractedCoordinates = []
    currentIndex = 0

    for idx, point in enumerate(resultant.pose_landmarks.landmark):
        while currentIndex < len(targetjoints):
            if targetjoints[currentIndex] < idx:
                currentIndex += 1
            elif targetjoints[currentIndex] > idx:
                break
            else:
                # extractedCoordinates.append([point.x, point.y])
                extractedCoordinates.append(point.x)
                extractedCoordinates.append(point.y)

                currentIndex += 1

    if display:

        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image_pose[:, :, ::-1]);
        plt.title("Input Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(original_image[:, :, ::-1]);
        plt.title("Pose detected Image");
        plt.axis('off');
        plt.show()

    else:
        return extractedCoordinates


def jointIDtoXYindex(id):
    return (id*2)-1, id*2
def videoToDf(video,df ,label="" ,id=""):
    frames = VideoPathToFrames(video)

    for frame in frames:
        result = detectPose(frame,draw=False, display=False)
        if result is not None:
          



            
            



            if label != "":
                result.append(label)
            if id != "":
                result.append(id)

            
 

            #tooooo slow
            # df = pd.concat([df, pd.Series(result, index=df.columns[:len(result)])])
            # decrapted
            df = df.append(pd.Series(result, index=df.columns[:len(result)]), ignore_index=True,)   
            #  
    return df      



def findAngle(x1, y1, x2, y2):
    theta = np.acos( (y2 -y1)*(-y1) / (np.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/np.pi)*theta
    return degree
        
    

def rootToXls(rootPath):
    output=rootPath+'.csv'
    dflst = []
    j = 0
    for i in range(0, (33 * 2) ):
        lable = ""
        if i % 2 == 0:
            lable += 'X'
            lable += str(j)

        else:
            lable += 'Y'
            lable += str(j)
            j += 1

        dflst.append(lable)

        # dflst.append('L_Shoulder_Elbow_angle')
        # dflst.append('R_Shoulder_Elbow_angle')
        # dflst.append('L_Elbow_Wrist_angle')
        # dflst.append('R_Elbow_Wrist_angle')

        # dflst.append('L_Waist_Knee_angle')
        # dflst.append('R_Waist_Knee_angle')
        # dflst.append('L_Knee_Foot_angle')
        # dflst.append('R_Knee_Foot_angle')


    
    #uncomment block for feature extraction
    # features_list = ["min","max","mean","std","energy"
    #     ,"rms","variance","skewness","kurtosis","median","mode","range"]
    # print(len(features_list))
    # for i,f in  enumerate( features_list):
    #     lable = ""
    #     lable+="X_"
    #     lable+=f
    #     dflst.append(lable)
    #     lable = ""
    #     lable += "Y_"
    #     lable += f
    #     dflst.append(lable)



    dflst.append("label")
    dflst.append("videoID")
    df = pd.DataFrame(columns = dflst)
    videoid =0
    actions = os.listdir(rootPath)
    AllLandmarks = []
    for action in actions:
        print(action)
        videos = os.listdir(os.path.join(rootPath,action))
        for video in videos:
            df = videoToDf(video=os.path.join(rootPath,action,video),label = action, id = videoid,df=df)
            videoid += 1
    
    print(df)
    df.to_csv(output, index=False)




if __name__ == "__main__":
    rootToXls('test')




