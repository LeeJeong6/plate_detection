import cv2
from ultralytics import YOLO
import os
from typing import List, Dict

def process_video(video_path: str) -> List[Dict[str, List[float]]]:
    """
    주어진 영상 파일을 프레임별로 처리하고, YOLO 모델을 사용하여 객체를 탐지한 후
    각 객체의 xywh 좌표와 (2배, 4배 키운) width, height 값을 반환합니다.

    Args:
        video_path (str): 처리할 영상 파일 경로입니다. 예) '/path/to/video.mp4'

    Returns:
        List[Dict[str, List[float]]]: 각 프레임마다 탐지된 객체의 xywh 좌표와 (2배, 8배 키운) 
                                      width, height 값을 담은 딕셔너리의 리스트를 반환합니다.
                                      예) [{'xywh': [x_center, y_center, width, height], 'confidence': [0.99]}, ...]
    """
   
    model = YOLO('/mnt/hdd_6tb/bill0914/processed_quarter/runs/detect/train2-4분할+resize+DA2/weights/best.pt')
    save_dir = '/mnt/hdd_6tb/bill0914/byte-tracking/result_frame'
    ori_save_dir = "/mnt/hdd_6tb/bill0914/byte-tracking/original_frame"
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    detection_results = []  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝까지 읽었으면 종료
        frame_idx += 1
        
        results = model(frame)  
        frame_result = {'xywh': [], 'confidence': []}  # 프레임별 결과를 담을 딕셔너리
        
        for result in results[0].boxes:
            xywh = result.xywh[0].tolist()  
            conf = result.conf[0].item()  
            xywh[2] *= 2  
            xywh[3] *= 8  
            
            frame_result['xywh'].append(xywh)  # xywh 좌표 
            frame_result['confidence'].append(conf)  # confidence 값
        
        detection_results.append(frame_result) 
        annotated_frame = results[0].plot()  # 결과를 포함한 이미지를 가져옴
        
        # 저장할 파일 경로 지정
        save_path = os.path.join(save_dir, f"frame_{frame_idx}.jpg")
        ori_save_path = os.path.join(ori_save_dir, f"frame_{frame_idx}.jpg")
        # 객체가 표시된 프레임을 이미지로 저장
        cv2.imwrite(save_path, annotated_frame)  # 결과 이미지 저장
        cv2.imwrite(ori_save_path, frame) 
    cap.release()
    cv2.destroyAllWindows()

    return detection_results  