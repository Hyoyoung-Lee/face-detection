# < 실시간 영상 → 얼굴 검출 → 랜드마크 → 위치 계산 → 오버레이 합성 >

# 변수 소개
  # 1. 프레임 계열 (영상)
    # frame	원본 영상 프레임
    # resized_frame	연산용 축소 프레임
    # base_frame	overlay 기준 프레임
    # landmark_frame	디버깅용
    # output_frame	최종 결과
  # 2. 얼굴 관련
    # faces	: 얼굴 bounding box
    # shape_2d	68 랜드마크 좌표
    # center_x, center_y	얼굴 중심
    # face_size	얼굴 크기
    # face_sizes	smoothing buffer
    # face_roi	다음 프레임 탐색 영역
  # 3. 오버레이 관련 (이미지) 
    # overlay_img	PNG 스티커
    # overlay_size	얼굴 크기 기반 resize
    # mask	투명도 마스크
    # roi	합성 대상 영역
    # bg_part	배경
    # fg_part	스티커
    # combined	합성 결과

import cv2, dlib, sys
import numpy as np


##############초기세팅시작##############
# 1. 모델 로딩
# 2. 영상 준비
# 3. 오버레이 이미지 준비

scaler = 0.3

# 얼굴 탐지 & 랜드마크 추출
# 1) 얼굴 탐지 모델 load
  # 얼굴이냐 아니냐를 탐지
  # HOG + SVM 기반 모델
    # HOG : 이미지에서 특징점 추출. 얼굴의 윤곽선, 눈, 코, 입 등과 같은 특징을 감지하는 데 사용
    # SVM : HOG로 추출된 특징을 기반으로 얼굴인지 아닌지 분류하는 머신러닝 모델 
detector = dlib.get_frontal_face_detector()

# 2) 얼굴 랜드마크 (68개 특징점) 추출 모델 load
  # 트리 기반 회귀 모델
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 영상 load
cap = cv2.VideoCapture('samples/girl.mp4')

# 오버레이 이미지 load
# 이걸 해서 PNG의 경우 → BGRA (4채널) 그대로 로딩됨
# cv2.IMREAD_UNCHANGED 옵션 없으면 배경이 이상하게 나옴. 투명 이미지 처리 시 필수 옵션
overlay_img = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

##############초기세팅끝##############


##############함수정의시작##############

# 배경 이미지 위에 투명 PNG 이미지를 합성하는 함수
  # 아래 호출되는 부분 보면 프레임마다 호출
  # 추후 호출 참고 : apply_overlay(base_frame, overlay_img, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))
def apply_overlay(base_frame, overlay_img, center_x, center_y, overlay_size=None):
  
  ##############영상시작##############
  # 원본 영상 복사
  output_frame = base_frame.copy()
  #print(output_frame.shape)

  # 오버레이와 채널 수 맞추기 위해, 영상을 4채널(BGRA)로 변환
    # 왜? => 영상은 보통 3채널(BGR), overlay는 4채널(BGRA) -> 채널 수 맞추기 위해 alpha 추가
  if output_frame.shape[2] == 3:
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2BGRA)
    #print(output_frame.shape)

  ##############영상끝##############



  ##############이미지시작##############

  # 오버레이 이미지를 원하는 크기로 변경
    # 위 변수 중 overlay_img 는 고정 이미지, overlay_size는 얼굴 크기에 맞춰서 매 프레임마다 달라지는 값
    # 얼굴 크기는 매 프레임마다 달라질 수 있기 때문에, 오버레이 이미지도 그에 맞게 크기 조절 필요
    # copy 이유 : Resize는 in-place 연산이라 원본 이미지 변경될 수 있음. 다음 프레임에서 계속 재사용해야하므로, copy
    # cv2.resize(src, dsize)
  if overlay_size is not None:
    overlay_img = cv2.resize(overlay_img.copy(), overlay_size)

  # 이미지 채널 분리 (B, G, R, Alpha) -> 그러고나서 투명도 마스크 생성
    # RGBA 중 A(Alpha) → 투명도 마스크로 사용
    # 분리 이유 : 색 정보(BGR)와 투명도(Alpha)를 분리해야 “어디를 덮고 / 어디를 남길지” 결정 가능
    # PNG 가장자리 보면 톱니(계단 현상) 있어서 medianBlur로 부드럽게 처리
    # 5 사용 이유 : 가장자리 부드럽게 처리하는 데에 적당한 커널 크기. 반드시 홀수여야 하고, 1보다 커야 함
  # mask => “투명도를 나타내는 지도(= 어디를 붙일지에 대한 마스크)”
    # 투명한 부분만 뗀 그 투명 부분도 아니고, 투명한 걸 제거한 결과 이미지도 아님
    # mask = “여기(255)는 overlay 붙여라” “여기(0)는 붙이지 마라”
  b, g, r, a = cv2.split(overlay_img)
  mask = cv2.medianBlur(a, 5)

  # 오버레이 이미지 크기
    # 높이, 너비, 채널 수인데 여기서 채널 수는 무시.
  h, w, _ = overlay_img.shape

  ##############이미지끝##############


  ##############영상시작##############

  # 얼굴 위치 영역 추출. 추출하고 난 배경이 아니라 그 얼굴 영역.(현재 사각형)
    # 위에서 오버레이 이미지 크기로 구한 것 기준으로.
    # roi : Region of Interest. 관심 영역. / overlay를 얼굴 중심 기준으로 배치 위해 사용
    # numpy 슬라이싱. img[y1:y2, x1:x2] 여기서 y1~y2 : 세로 범위, x1~x2 : 가로 범위.
    # 여기서는 얼굴 중심(x, y) 기준으로 w, h 크기만큼 사각형 영역 추출
  # roi 따로 쓰는 이유 : 전체 이미지 처리 안 하고, 필요한 부분만 처리
  roi = output_frame[int(center_y-h/2):int(center_y+h/2), int(center_x-w/2):int(center_x+w/2)]

  ##############영상끝##############


  # 영상 영역과 오버레이 영역 분리
    # 영상 부분
      # cv2.bitwise_and(src1, src2, mask=mask) : mask 부분만 남기기 (나머지는 제거)
      # cv2.bitwise_not(mask) : mask 반전 (덮이는 부분 제거 위해)
      # 얼굴 영역(roi)에서 스티커가 들어갈 부분만 지우고, 나머지 배경만 남기는 작업
    # bitwise_and (src1, src2, mask) : src1과 src2의 bitwise AND 연산을 수행 => 즉, 두 이미지를 겹쳐서 둘 다 값이 있는 부분만 남김 => 즉, 공통 영역만 남김
  bg_part = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    # 오버레이 부분 (덮일 부분만 남김)
      # 스티커의 투명한 부분은 제거하고 실제 스티커만 남김
      # mask의 255인 부분(스티커가 있는 부분)만 남기고, 나머지는 제거하는 작업
  fg_part = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)


  # combined 된 결과 = (배경 * (1-mask)) + (오버레이 * mask)
  # 즉, 오버레이 제거된 배경 (전체 배경이 아니고 얼굴의 오버레이 사이즈영상 [=> roi] 에서 스티커 부분 제외된 것)
  #   + 오버레이 스티커 불투명한 부분만 합성
  #   => 빈공간 + 스티커 합치기 / 겹치는 부분이 없어 add로 충분
  combined = cv2.add(bg_part, fg_part)


  # 원본 영상에 다시 삽입 (ROI 영역 업데이트됨!)
  # x, y : 얼굴 중심 좌표 / w, h : 오버레이 이미지 크기
  # 얼굴 중심(x, y)을 기준으로 overlay 크기만큼 사각형 영역 부분이 교체됨
    # 여기서는 픽셀값 충돌이 없는가?
      # -> combined이 이미 원본 배경을 포함하고 있으므로, 그대로 덮어써도 배경이 사라지거나 충돌하는 일이 없음 단순히 "같은 자리에 완성본을 교체"하는 것
      # -> 만약 fg_part만 덮어썼다면 배경이 날아갔겠지만, bg_part + fg_part로 합친 combined을 넣기 때문에 문제없음
  output_frame[int(center_y-h/2):int(center_y+h/2), int(center_x-w/2):int(center_x+w/2)] = combined

  # 다시 BGR 3채널로 변환 (최종 출력은 화면 출력 및 영상 저장이라 BGR 3채널만 필요)
  return cv2.cvtColor(output_frame, cv2.COLOR_BGRA2BGR)

##############함수정의끝##############



##############while루프시작##############
# 얼굴 위치와 크기 저장 변수
face_roi = []
face_sizes = []

# loop
# 1초에 보통 30프레임, 함수가 프레임마다 호출돼서 print()도 프레임마다 실행
while True:
  # 초반에 video 가져온 것을 읽음. read()로 가져온 것은 프레임 단위의 이미지. ret은 프레임 읽기 성공 여부(True/False), frame은 실제 프레임 이미지
  ret, frame = cap.read()
  if not ret:
    break

  # 1) 영상 resize
    # 위에 scaler = 0.3 으로 설정되어 있어서, 원본 영상의 30% 크기로 축소
  resized_frame = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))
  
  # 2) 기준 원본 (Resize 된 것 그대로)
  base_frame = resized_frame.copy()

  # 3) 랜드마크 시각화용
  landmark_frame = resized_frame.copy()

  # 4) 최종 결과 (오버레이 적용)
  output_frame = base_frame.copy()


  # 얼굴 탐지
    # 처음 (face_roi 길이가 0)에는 전체 탐색
    # 이후, roi 부분만 탐색하여 속도 증가 (else 분기로.)
  if len(face_roi) == 0:
    faces = detector(resized_frame)
    #cv2.imshow('resize', resized_frame)

  # image[세로(y), 가로(x)]
  # 이미지에서 특정 사각형 영역(ROI)을 잘라낸 후, 그 영역에서 얼굴 탐지 수행
  else:
    roi_frame = resized_frame[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    #cv2.imshow('roi', roi_frame)
    faces = detector(roi_frame)

  # no faces
  if len(faces) == 0:
    print('no faces!')

  # 위에 Faces가 비어있으면 (길이가 0이면) 반복 대상이 없어서 for문은 실행 X
  for face in faces:

    # 랜드마크 추출
      # shape_2d : 68개 랜드마크 좌표. 얼굴의 특징점(눈, 코, 입 등)의 위치를 나타내는 68개의 좌표로 구성된 배열
      # dlib_shape : dlib의 shape_predictor가 반환하는 객체로, 얼굴 랜드마크의 좌표 정보를 담고 있음.
        # 이 객체에서 .parts() 메서드를 사용하여 각 랜드마크의 좌표를 추출할 수 있음

    # 1) 전체 프레임 기준
      # 이 분기에서는 shape_2d = "전체 프레임 기준 좌표"
      # p.x, p.y 는 랜드마크 좌표를 뜻함
      # face_roi : [y1, y2, x1, x2] 형태의 배열로, 다음 프레임에서 탐색할 영역을 정의
    if len(face_roi) == 0:
      dlib_shape = predictor(resized_frame, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # 2) ROI 프레임 기준
      # 이 분기에서는 ROI의 (0,0) = 원래 이미지의 (x1, y1)
      # 따라서 원래 p.x, p.y 에서 ROI 시작좌표(x1, y1) 만큼 움직여줘야 함
      # 따라서 이 분기에서의 shape_2d = "ROI 프레임 기준 좌표" 
      # face_roi : [y1, y2, x1, x2] 형태의 배열로, 다음 프레임에서 탐색할 영역을 정의
      # 아래 face_roi 정의 보면, face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])   
      # face_roi = [y1, y2, x1, x2]
      #              0   1   2   3 => 각각 위, 아래, 왼쪽, 오른쪽 좌표
      # ROI의 시작 x 좌표 => face_roi[2]
        # -> ROI 기준 랜드마크 좌표는 p.x + face_roi[2] (x1) 만큼 이동
      # ROI의 시작 y 좌표 => face_roi[0]
        # => ROI 기준 랜드마크 좌표는 p.y + face_roi[0] (y1) 만큼 이동
    else:
      dlib_shape = predictor(roi_frame, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

    # 여기까지 shape_2d는 이런 모양. 
      # shape_2d =
      # [
      #   [x1, y1],
      #   [x2, y2],
      #   ...
      #   [x68, y68]
      # ]


    # 랜드마크 시각화 : 얼굴이 제대로 인식이 되는지 확인 (오버레이를 정확하게 붙이기 위해)
      # 각 점마다 circle 찍음
      # tuple(s) : s는 shape_2d의 각 행(각 랜드마크 좌표) => (x, y) 형태로 튜플 변환
      # radius=1 : 원의 반지름 1 픽셀
      # color=(255, 255, 255) : 흰색
      # LINE_AA : Anti-Aliasing 적용해서 선을 부드럽게 그리는 옵션 (계단 현상 줄이기 위해)
    for s in shape_2d:
      cv2.circle(landmark_frame, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴 중심 좌표 : 스티커를 어디 중심으로 붙일지 결정하기 위함
      # x 평균 = (x1 + x2 + ... + x68) / 68
      # y 평균 = (y1 + y2 + ... + y68) / 68
    center_x, center_y = np.mean(shape_2d, axis=0).astype(int)

    # 얼굴 경계 : 얼굴 영역 크기 계산
      # min_coords = [min_x, min_y]
      # max_coords = [max_x, max_y]
    # 아래 face_size와의 차이 : face_size는 얼굴 크기 계산해서 버퍼에 저장하는 값, min_coords/max_coords는 얼굴 영역 시각화 위해 계산하는 값
    min_coords = np.min(shape_2d, axis=0)
    max_coords = np.max(shape_2d, axis=0)

    # bounding box 확인용
      # color=(255, 0, 0) : 파란색
      # 위에서 계산한 min_coords, max_coords를 시각화해서 얼굴 영역이 제대로 잡히는지 확인
    cv2.circle(landmark_frame, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(landmark_frame, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)


    # 얼굴 크기 계산 (핵심 로직)
      # max 사용 이유 : 얼굴은 비율이 일정하지 않아서, 작은 쪽 기준으로 하면 스티커가 잘릴 수 있음. 큰 쪽이 안정적
      # face_sizes : 과거 버퍼 / face_size : 현재 프레임 값
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)

    # 버퍼 최근 10개만 유지
    # 이유 : 얼굴 크기 변화에 대한 안정화. 프레임마다 얼굴 크기가 미세하게 변할 수 있는데, 최근값 평균을 통해 안정화 효과를 줌
    if len(face_sizes) > 10:
      del face_sizes[0]
    # 버퍼 평균 내서 얼굴 크기 안정화
    # 1.8 : 얼굴보다 크게 스티커 씌우기. 얼굴크기 계산해서 이만큼 크게~
    mean_face_size = int(np.mean(face_sizes) * 1.8)

    # ROI 배열 업데이트 (다음 루프에서 얼굴 탐지할 때, 전체 프레임이 아니라 이 영역에서 탐지하도록)
      # face_roi : [y1, y2, x1, x2] 형태의 배열로, 다음 프레임에서 탐색할 영역을 정의
      # 얼굴 중심 기준으로 face_size 만큼 여유 공간을 둔 사각형 영역
      # 영상 크기보다 ROI가 커지는 경우 방지 (음수나 영상 크기보다 큰 값이 들어가는 경우 방지)
      # clip : 범위 벗어나는 값은 가장자리 값으로 고정
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

    # overlay 적용
    output_frame = apply_overlay(base_frame, overlay_img, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))

##############while루프끝##############

  # visualize
  cv2.imshow('original', base_frame)
  cv2.imshow('facial landmarks', landmark_frame)
  cv2.imshow('result', output_frame)
  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)
