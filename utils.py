import numpy as np

def createCircularMask(radius):
    X, Y, Z = np.ogrid[:radius, :radius, :radius]
    dist_from_center = np.sqrt(X**2 + Y**2 + Z**2)
    mask = dist_from_center <= radius
    mask = np.array(mask, dtype=np.uint8)
    return mask

def searchPoint(img, center, radius=8):
    """
    img: 3d array, M * N * K image
    center: 1 * 3 array, [x_A, y_A, z_A] the search center
    radius: float value, search radius
    return: boolean value, if there exists a point in the search range in the image
    """
    m, n, k = center
    mask = createCircularMask(radius)
    crop = img[m:m+radius, n:n+radius, k:k+radius].copy()
    crop = np.resize(crop, [radius, radius, radius])
    if crop[mask].sum() > 0:
        return True
    else:
        return False
    
def computeNeuronalTruePositive(groundtruth, prediction, tolerance=8):
    """
    groudthrth: numpy array, M * N * K
    prediction: numpy array, M * N * K
    return: the true positive number of prediction
    """
    result = 0
    [M, N, K] = prediction.shape
    for i in range(M):
        for j in range(N):
            for p in range(K):
                position = [i, j, p]
                if prediction[i, j, p] == 1 and searchPoint(groundtruth, position):
                    result += 1
    return result
