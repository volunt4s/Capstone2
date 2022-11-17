import numpy as np

if __name__ == "__main__":
    # numpy 데이터 로딩하기
    data_x = np.load("aug_x.npy")
    data_y = np.load("aug_y.npy")
    # 데이터 shape 출력
    print(f"data_x shape : {data_x.shape}")
    print(f"data_y shape : {data_y.shape}")
    # 결함 타입별 개수 출력
    faulty_case = np.unique(data_y)

    for fault in faulty_case:
        print(f"{fault} : {len(data_y[data_y==fault])}")