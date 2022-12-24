import cv2
import numpy as np
import os

#
# Выполнили Черняев Андрей и Сергей Карманов
#

# При желании можно проверить на других картинках

# фотография сома
#img_file = "catfish.png"
#img_fragment_file = "fragment.png"

# постер фильма
# img_file = "drive.png"
# img_fragment_file = "drivefrg.png"

# котики
img_file = "specat.jpg"
img_fragment_file = "specatf.jpg"

# картина
# img_file = "moremore.png"
# img_fragment_file = "moremoref.png"

# Подготовка данных
img = cv2.imread(img_file, cv2.IMREAD_COLOR)
img_fragment = cv2.imread(img_fragment_file, cv2.IMREAD_COLOR)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("image", img_fragment)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Константы
img_height = len(img)
img_width = len(img[0])
img_fragment_height = len(img_fragment)
img_fragment_width = len(img_fragment[0])
channel_count = 3
cluster_count = 8

params = np.array(img, dtype=float)
# Подсчет длины вектора RGB
img_vector_len = [[np.sqrt(np.sum(np.square(params[i, j, :]))) for j in range(img_width)] for i in range(img_height)]
# Нормирование параметров
norm_params = np.array([[[(img[k][i][j] / img_vector_len[k][i]) if img_vector_len[k][i] != 0 else 0 for j in range(channel_count)] for i in range(img_width)] for k in range(img_height)])
params = np.array(img_fragment, dtype=float)
img_fragment_vector_len = [[np.sqrt(np.sum(np.square(params[i, j, :]))) for j in range(img_fragment_width)] for i in range(img_fragment_height)]
norm_fragment_params = np.array([[[(img_fragment[k][i][j] / img_fragment_vector_len[k][i]) if img_fragment_vector_len[k][i] != 0 else 0 for j in range(channel_count)] for i in range(img_fragment_width)] for k in range(img_fragment_height)])

# Задаются веса для 3 входных каналов RGB
weights = np.random.random((channel_count, cluster_count))
weights_vector_len = [np.sqrt(np.sum(np.square(weights[:, i]))) for i in range(cluster_count)]
norm_weights = [[weights[i][j] / weights_vector_len[j] for j in range(cluster_count)] for i in range(channel_count)]

# Обучение
print("Обучение сети")
for k in range(img_height):
    for i in range(img_width):
        # Получение индекса "победителя"
        output_index = np.argmax(np.dot(norm_params[k][i], norm_weights))
        # Корректировка весов
        for j in range(channel_count):
            norm_weights[j][output_index] = norm_weights[j][output_index] + 0.5 * (
                    norm_params[k][i][j] - norm_weights[j][output_index])

resolution = img_fragment_height * img_fragment_width

clusters = np.array([[np.argmax(np.dot(norm_fragment_params[j][i], norm_weights)) for i in range(img_fragment_width)] for j in range(img_fragment_height)], dtype=int)
clusters = np.unique(clusters, return_counts=True)
clusters = np.around(clusters[1] / resolution, 2)

buffer = np.array([[np.argmax(np.dot(norm_params[k][i], norm_weights)) for i in range(img_width)] for k in range(img_height)], dtype=int)

flag = False
for k in range(img_height - img_fragment_height):
    os.system('cls||clear')
    print("Изображение проанализировано на ", (k+1)/(img_height-img_fragment_height) * 100, "%")
    for i in range(img_width - img_fragment_width):
        temp = np.unique(buffer[k:k + img_fragment_height, i:i + img_fragment_width], return_counts=True)
        temp = np.around(temp[1] / resolution, 2)
        if np.array_equal(clusters, temp):
            cv2.rectangle(img, (i, k), (i + img_fragment_width, k + img_fragment_height), (0, 0, 255), 2)
            flag = True
            break
    if flag:
        break

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

