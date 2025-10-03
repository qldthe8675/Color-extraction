import cv2
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 단일 이미지에서 훈련 데이터 수집
def collect_training_data(image_path, lower_red1, upper_red1, lower_red2, upper_red2, save_mask=False, output_dir='masks'):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 이미지를 로드할 수 없습니다. 파일이 유효한 이미지인지 확인하세요.")
        return None, None

    # HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 마스크 저장 (디버깅용)
    if save_mask:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        mask_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_mask.png')
        cv2.imwrite(mask_path, mask)
        print(f"마스크가 {mask_path}에 저장되었습니다.")

    # 결과 이미지 생성
    hsv_flat=hsv.reshape(-1, 3).astype(np.float32)
    hsv_flat[:,0] /= 179.0  # Hue 정규화
    hsv_flat[:,1] /= 255.0  # Saturation 정규화
    hsv_flat[:,2] /= 255.0  # Value 정규화
    labels=(mask.reshape(-1) / 255).astype(np.float32)

    return hsv_flat, labels

# 다중 이미지 데이터 수집
def collect_training_data_from_directory(directory, lower_red1, upper_red1, lower_red2, upper_red2):
    hsv_list = []
    labels_list = []
    training_dir = r'C://Users//user//Desktop//CVDcoding//training_images'  
    # 훈련 이미지 디렉토리

    if not os.path.exists(training_dir):
        print(f"Error: 디렉토리가 {training_dir}에 존재하지 않습니다.")
        return None, None
    
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            hsv, labels = collect_training_data(image_path, lower_red1, upper_red1, lower_red2, upper_red2, save_mask=True)
            if hsv is not None:
                hsv_list.append(hsv)
                labels_list.append(labels)
    if hsv_list:
        hsv_data = np.vstack(hsv_list)
        labels_data = np.hstack(labels_list)
        return hsv_data, labels_data
    else:
        print("Warning: 처리된 이미지가 없습니다.")
        return None, None
    
# 데이터 증강
def augment_data(hsv_data, labels, num_augmentations=5):
    augmented_hsv = []
    augmented_labels = []
    for i in range(len(hsv_data)):
        h, s, v = hsv_data[i]
        label = labels[i]
        augmented_hsv.append([h, s, v])
        augmented_labels.append(label)
        for _ in range(num_augmentations):
            h_aug = (h + np.random.uniform(-0.028, 0.028)) % 1.0  # Hue ±5/179
            s_aug = np.clip(s * np.random.uniform(0.9, 1.1), 0, 1.0)
            v_aug = np.clip(v * np.random.uniform(0.8, 1.2), 0, 1.0)
            augmented_hsv.append([h_aug, s_aug, v_aug])
            augmented_labels.append(label)
    return np.array(augmented_hsv, dtype=np.float32), np.array(augmented_labels, dtype=np.float32)

# ANN 훈련
def train_ann(hsv_data, labels, model_path='mlp_model.xml'):
    mlp = cv2.ml.ANN_MLP_create()
    mlp.setLayerSizes(np.array([3, 20, 10, 1]))
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    mlp.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 300, 0.001))
    
    mlp.train(hsv_data, cv2.ml.ROW_SAMPLE, labels.reshape(-1, 1))
    mlp.save(model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")
    
    return mlp

# 실시간 비디오 처리
def process_video_with_ann(model_path='mlp_model.xml'):
    
    # 트랙바 창 생성
    cv2.namedWindow("Trackbars")

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 카메라 인덱스에 따라 변경
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        mlp = cv2.ml.ANN_MLP_load(model_path)
    except:
        print(f"Error: 모델 파일 {model_path}을 로드할 수 없습니다. 먼저 모델을 훈련시키세요.")
        return
    
    while True:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            print("Error: 프레임을 읽을 수 없습니다.")
            break
        
        ori_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_flat = hsv.reshape(-1, 3).astype(np.float32)

        hsv_flat[:,0] /= 179.0
        hsv_flat[:,1] /= 255.0
        hsv_flat[:,2] /= 255.0
        _, predictions = mlp.predict(hsv_flat)
        mask = (predictions > 0.8).reshape(frame.shape[:2]).astype(np.uint8) * 255

        # 적색 픽셀을 흰색으로, 나머지는 검정색으로 표시
        masked_frame = np.zeros_like(ori_frame)
        masked_frame[mask == 255] = [255, 255, 255]

        # 프레임당 처리 시간과 FPS 계산
        time_elapsed=time.time() - start_time
        fps=1/time_elapsed if time_elapsed > 0 else 0

        cv2.putText(masked_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(masked_frame, f"Time: {time_elapsed*1000:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Original Frame", ori_frame)
        cv2.imshow("Masked Frame", masked_frame)
        
        if cv2.waitKey(33) == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

# 가중치 분포 시각화
def plot_weights_distribution(model_path='mlp_model.xml'):
    try:
        tree = ET.parse(model_path)
        root = tree.getroot()
        weight_elements = root.findall('opencv_ml_ann_mlp/weights/_')

        num_layers=len(weight_elements)
        if num_layers == 0:
            print("Error: 가중치 정보가 없습니다.")
            return
        
        fig, axes=plt.subplots(num_layers, 1, figsize=(8 * num_layers, 6))
        if num_layers == 1:
            axes=[axes]  # 단일 레이어의 경우 리스트로 변환

        for i, (weight_array, ax) in enumerate(zip(weight_elements, axes)):
            print(f"가중치 분포 figure 생성: 레이어 {i+1}")
            weight_str = weight_array.text.strip()
            weights = np.fromstring(weight_str, sep=' ')
            ax.hist(weights, bins=50, edgecolor='black')
            ax.set_title(f'Layer {i+1} Weights Distribution')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)  # 잠시 대기하여 그래프를 업데이트

    except Exception as e:
        print(f"XML 파싱 오류: {e}")

# HSV 결정 경계 시각화
def plot_hsv_decision_boundary_with_data(mlp, hsv_train, labels_train, V_fixed=128, H_step=1, S_step=5):
    V_norm = V_fixed / 255.0
    H_values = np.arange(0, 180, H_step)
    S_values = np.arange(0, 256, S_step)
    H_grid, S_grid = np.meshgrid(H_values, S_values)
    H_flat = H_grid.ravel()
    S_flat = S_grid.ravel()
    V_flat = np.full_like(H_flat, V_norm)
    hsv_points = np.vstack([H_flat / 179.0, S_flat / 255.0, V_flat]).T.astype(np.float32)
    _, predictions = mlp.predict(hsv_points)
    predictions = predictions.reshape(H_grid.shape)
    hsv_train_denorm = hsv_train.copy()
    hsv_train_denorm[:, 0] *= 179.0
    hsv_train_denorm[:, 1] *= 255.0
    hsv_train_denorm[:, 2] *= 255.0
    mask = np.abs(hsv_train_denorm[:, 2] - V_fixed) < 10
    hsv_subset = hsv_train_denorm[mask]
    labels_subset = labels_train[mask]
    plt.figure(figsize=(10, 8))
    plt.contourf(H_values, S_values, predictions, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='Prediction (0: Non-Red, 1: Red)')
    red_points = hsv_subset[labels_subset == 1]
    non_red_points = hsv_subset[labels_subset == 0]
    if len(red_points) > 0:
        plt.scatter(red_points[:, 0], red_points[:, 1], c='red', label='Red Pixels', alpha=0.5, s=10)
    if len(non_red_points) > 0:
        plt.scatter(non_red_points[:, 0], non_red_points[:, 1], c='blue', label='Non-Red Pixels', alpha=0.5, s=10)
    plt.xlabel('Hue (0-179)')
    plt.ylabel('Saturation (0-255)')
    plt.title(f'ANN Decision Boundary with Training Data at V={V_fixed}')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.001)  # 잠시 대기하여 그래프를 업데이트

# 메인 실행
if __name__ == "__main__":
    # 적색 범위 설정
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # 훈련 데이터 수집
    training_dir = r'C:/Users//user//Desktop//CVDcoding//training_images'  # 훈련 이미지 디렉토리
    hsv_data, labels = collect_training_data_from_directory(training_dir, lower_red1, upper_red1, lower_red2, upper_red2)
    
     # 디렉토리 생성 또는 확인
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        print(f"디렉토리 {training_dir}를 생성했습니다. 이미지를 추가한 후 다시 실행하세요.")
        exit()

    if hsv_data is not None:
        hsv_train, hsv_test, labels_train, labels_test = train_test_split(hsv_data, labels, test_size=0.2, random_state=42)
        # 데이터 균형 조정
        red_indices = np.where(labels_train == 1)[0]
        non_red_indices = np.where(labels_train == 0)[0]
        num_red = len(red_indices)

        if num_red == 0:
            print("Error: No red pixels in training data.")
            
        else:
            # 디버깅: 데이터 크기 확인
            print(f"hsv_train shape: {hsv_train.shape}, labels_train shape: {labels_train.shape}")
            print(f"red_indices count: {len(red_indices)}, non_red_indices count: {len(non_red_indices)}")

            if len(non_red_indices) > num_red:
                sampled_non_red_indices = np.random.choice(non_red_indices, size=num_red, replace=False)
            else:
                sampled_non_red_indices = non_red_indices

            selected_indices = np.concatenate([red_indices, sampled_non_red_indices])

            # 인덱스 유효성 검증
            max_index = hsv_train.shape[0] - 1
            if np.any(selected_indices > max_index):
                print(f"Error: selected_indices contains invalid indices. Max valid index: {max_index}")
                selected_indices = selected_indices[selected_indices <= max_index]

            np.random.shuffle(selected_indices)

            # 데이터 선택
            hsv_selected = hsv_train[selected_indices]
            labels_selected = labels_train[selected_indices]

            max_samples = 1000  # 시스템 성능에 따라 조정
            if len(hsv_selected) > max_samples:
                indices = np.random.choice(len(hsv_selected), max_samples, replace=False)
                hsv_selected = hsv_selected[indices]
                labels_selected = labels_selected[indices]
            
            # 데이터 증강
            hsv_aug, labels_aug = augment_data(hsv_selected, labels_selected, num_augmentations=5)
            # print("debugging_1")
            
            # ANN 훈련
            mlp = train_ann(hsv_aug, labels_aug)
            
            # 테스트 데이터로 평가
            _, predictions_test = mlp.predict(hsv_test)
            predictions_test = (predictions_test > 0.8).astype(int).flatten()
            accuracy = accuracy_score(labels_test, predictions_test)
            precision = precision_score(labels_test, predictions_test)
            recall = recall_score(labels_test, predictions_test)
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")

            # 실시간 비디오 처리
            process_video_with_ann()