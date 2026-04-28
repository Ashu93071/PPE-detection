from ultralytics import YOLO
import cv2

model_path=r'project\trained_model\train\weights\best.pt'

fined_model=YOLO(model_path)

# source_path=r'project\test_video2.mp4'
source_path="http://10.152.114.21:8080/video"

class PPEDetection:
    def __init__(self,model_path):
        self.model=YOLO(model_path)

    def process_video(self,source_path):
        cap=cv2.VideoCapture(source_path)
        if not cap.isOpened():
            # print(f"Error: Could not open video source {source_path}")
            return (f"Error: Could not open video source {source_path}")
        
        while cap.isOpened():
            response={}
            ret,frame=cap.read()
            if not ret:
                print(response)
                break

            #Run tracking
            result=self.model.track(frame,persist=True)
            annoted_frame=result[0].plot()
            for box in result[0].boxes:
                track_id=box.id
                class_id=int(box.cls[0])
                label=self.model.names[class_id]

   
                if label=='NO-Hardhat':
                    response['Track ID']=track_id
                    response['Alert']='Hardhat was not detected'
 
                elif label=='NO-Safety Vest':
                    response['Track ID']=track_id
                    response['Alert']='Safety vest was not detected'


            cv2.imshow('Annoted frame:',annoted_frame)
            
            if cv2.waitKey(1)==ord('q'):
                print(response)
                break
            
        cap.release()
        cv2.destroyAllWindows()    


if __name__=='__main__':
    detector=PPEDetection(model_path)
    detector.process_video(source_path)