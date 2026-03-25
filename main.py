# < 실시간 영상 → 얼굴 검출 → 랜드마크 → 위치 계산 → 오버레이 합성 >

import cv2, dlib, sys
import numpy as np



scaler = 0.3

# 얼굴 탐지 & 랜드마크 추출
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture('samples/girl.mp4')

# 오버레이 이미지 load -> PNG 파일 BGRA (4채널) 그대로 로딩됨
overlay_img = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)



# 배경 이미지 위에 투명 PNG 이미지를 합성하는 함수
def apply_overlay(base_frame, overlay_img, center_x, center_y, overlay_size=None):
  
  output_frame = base_frame.copy()

  # 영상은 보통 3채널(BGR), overlay는 4채널(BGRA)
  if output_frame.shape[2] == 3:
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2BGRA)
    #print(output_frame.shape)


  # 오버레이 이미지를 원하는 크기로 변경
    # overlay_size는 얼굴 크기에 맞춰서 매 프레임마다 달라지는 값
  if overlay_size is not None:
    overlay_img = cv2.resize(overlay_img.copy(), overlay_size)

  # 이미지 채널 분리 (B, G, R, Alpha) -> 투명도 마스크 생성
  b, g, r, a = cv2.split(overlay_img)
  mask = cv2.medianBlur(a, 5)

  # 오버레이 이미지 크기 => # 높이, 너비, 채널 수
  # roi 따로 쓰는 이유 : 전체 이미지 처리 안 하고, 필요한 부분만 처리
  # 얼굴 중심(x, y) 기준으로 w, h 크기만큼 사각형 영역 추출
  h, w, _ = overlay_img.shape
  roi = output_frame[int(center_y-h/2):int(center_y+h/2), int(center_x-w/2):int(center_x+w/2)]


  # 영상 얼굴 영역(roi)에서 스티커가 들어갈 부분만 지우고, 나머지 배경만 남기는 작업
  # bitwise_and (src1, src2, mask) => 즉, 두 이미지를 겹쳐서 공통 영역만 남김
  bg_part = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  # 실제 스티커만 남김
  fg_part = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)


  combined = cv2.add(bg_part, fg_part)
  output_frame[int(center_y-h/2):int(center_y+h/2), int(center_x-w/2):int(center_x+w/2)] = combined

  # 최종 출력 : 영상 출력이라 BGR 3채널만 필요
  return cv2.cvtColor(output_frame, cv2.COLOR_BGRA2BGR)





face_roi = []
face_sizes = []

while True:
  ret, frame = cap.read()
  if not ret:
    break

  resized_frame = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))
  base_frame = resized_frame.copy()
  landmark_frame = resized_frame.copy()
  output_frame = base_frame.copy()


  # 얼굴 탐지 : 처음 (face_roi 길이가 0)에는 전체 탐색 -> 이후 roi 부분만 탐색하여 속도 증가
  if len(face_roi) == 0:
    faces = detector(resized_frame)
  else:
    roi_frame = resized_frame[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    faces = detector(roi_frame)

  if len(faces) == 0:
    print('no faces!')

  # 랜드마크 추출
  for face in faces:

    # 전체 프레임 기준
    if len(face_roi) == 0:
      dlib_shape = predictor(resized_frame, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # ROI 프레임 기준
    else:
      dlib_shape = predictor(roi_frame, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])



    # 랜드마크 시각화
    for s in shape_2d:
      cv2.circle(landmark_frame, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴 중심 좌표
    center_x, center_y = np.mean(shape_2d, axis=0).astype(int)

    # 영역 크기 계산 및 시각화
    min_coords = np.min(shape_2d, axis=0)
    max_coords = np.max(shape_2d, axis=0)

    cv2.circle(landmark_frame, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(landmark_frame, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)



    # 스티커 크기 계산 위해 -> 얼굴 크기 계산
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)

    # 버퍼 최근 10개만 유지 : 얼굴 크기 변화에 대한 안정화 효과
    if len(face_sizes) > 10:
      del face_sizes[0]
    # 1.8 : 얼굴보다 크게 스티커 씌우기~
    mean_face_size = int(np.mean(face_sizes) * 1.8)

    # ROI 배열 업데이트 : 다음 루프에서 얼굴 탐지할 때, 전체 프레임이 아니라 이 영역에서 탐지하도록
      # 얼굴 중심 기준으로 face_size 만큼 여유 공간을 둔 사각형 영역
      # clip : 범위 벗어나는 값은 가장자리 값으로 고정
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

    output_frame = apply_overlay(base_frame, overlay_img, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))


  # visualize
  cv2.imshow('original', base_frame)
  cv2.imshow('facial landmarks', landmark_frame)
  cv2.imshow('result', output_frame)

  # 'q' 누르면 종료
  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)
