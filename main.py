import vi_quarter
import bytetrack 
import result_draw
import video_creator
video_path = 'original_video_path~.mp4'
frame_dir = 'path_your_saved_original_frames'
output_dir = 'path_your_saved_predicted_frames' 
def result_print(object_ids):
    '''
    프레임별로 탐지한 bbox를 동영상으로 만들어 완성합니다

    Args:
        object_ids (list) : 프레임에 대한 정보 (ID, xyxy, confidence) 

    Returns:
        None
    '''
    for frame_idx, frame_objects in enumerate(object_ids):
        print(f"Frame {frame_idx + 1}:")
        for obj in frame_objects:
            print(f"  ID: {obj['id']}, xyxy: {obj['xyxy']}, confidence: {obj['conf']}")
    for frame_idx, frame_objects in enumerate(object_ids):
   
        # result_draw 모듈의 함수를 호출하여 프레임에 ID, xyxy, confidence 그리기
        result_draw.draw_tracking_results(frame_idx, frame_objects, frame_dir, output_dir)
    video_creator.create_video_from_frames(output_dir,"path_your_saved_video_root.avi")   

if __name__ == '__main__':
    '''
    작은 객체를 탐지하기 위한 Trakcing 알고리즘
    가로를 n배 , 세로를 m배 키워서 프레임 간 객체의 IOU를 높게 키우는 원리
    
    '''



    object_ids = vi_quarter.process_video(video_path) 
    
    result_print(object_ids)
