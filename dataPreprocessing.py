import cv2
import os
import numpy as np

class dataPreprocessing():
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path

    def SetBoundary(self) -> None:
        for filename in os.listdir(self.folder_path):
            # 마스크 이미지 로드
            mask_image = cv2.imread(f'{self.folder_path}/{filename}', cv2.IMREAD_GRAYSCALE)

            # 배경과 객체 사이의 경계를 2로 만들어주기
            border_mask = np.zeros_like(mask_image)
            border_mask[np.where(mask_image == 0)] = 0
            border_mask[np.where(mask_image == 255)] = 255
            
            # 배경과 객체 사이의 경계를 확장하여 2로 설정
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
            dilated_mask = cv2.dilate(border_mask, kernel, iterations=1)
            
            boundary_mask = dilated_mask-border_mask

            # 배경
            border_mask[np.where(border_mask == 0)] = 3
            # 객체
            border_mask[np.where(border_mask == 255)] = 1
            # 경계
            border_mask[np.where(boundary_mask == 255)] = 2

            # 결과 마스크 이미지 저장
            cv2.imwrite(f'data/preprocessed_mask/{filename}', border_mask)

    def RemoveWhiteSpaceAtFileName(self) -> None:
        for filename in os.listdir(self.folder_path):
            old_img_path = f'{self.folder_path}/{filename}'
            new_img_path = old_img_path.replace(" ", "_")
            os.rename(old_img_path, new_img_path)
        


if __name__ == "__main__":  
    data_preprocessor = dataPreprocessing(folder_path='./data/img')
    data_preprocessor.RemoveWhiteSpaceAtFileName()