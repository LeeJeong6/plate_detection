# plate_detection
동영상에서 번호판과 같이 작은 개체를 tracking하기 위한 알고리즘입니다

<알고리즘>

1.동영상을 프레임 단위로 저장

2.각 프레임에서 탐지한 객체의 크기를 증가(default : 가로3배,세로6배) -> vi_quarter.py의 def detect_objects에서 조정 가능(조정한 값은 result_draw.py의 def draw_tracking_results 의 resize_factors에도 바꿔야함)

3.프레임 중간중간에 탐지를 못하는 문제, 작은 객체 탐지 불가 문제를 해결하기 위해 
테스트 데이터는 4분할되어 모델에 테스트되며, 7프레임 동안 정보를 기억하는 방법으로 끊기는 문제를 해결

4.동영상으로 합치기

main.py의 경로를 설정해야합니다

<video_path>:원본 비디오

<frame_dir>:원본 비디오의 프레임이 저장될 폴더

<output_dir>:모델이 예측한 프레임이 저장될 폴더

25번째 줄의 video_creator.create_video_from_frames(output_dir,동영상이 저장될 경로~.avi) #avi or mp4

