import cv2
import numpy as np
import time
import os

#
# Выполнили Черняев Андрей и Сергей Карманов
#


# При желании можно проверить на других картинках

# фотография сома
#img_file = "catfish.png"
#img_fragment_file = "fragment.png"

# постер фильма
#img_file = "drive.png"
#img_fragment_file = "drivefrg.png"

# котики
# img_file = "specat.jpg"
# img_fragment_file = "specatf.jpg"

# картина
img_file = "moremore.png"
img_fragment_file = "moremoref.png"

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

# Нормирование параметров
# Подсчет длины вектора RGB
# img_vector_lena = [
#     [(((img[i][j][0] ** 2) + (img[i][j][1] ** 2) + (img[i][j][2] ** 2)) ** 0.5) for j in range(img_width)] for i in
#     range(img_height)]
params = np.array(img, dtype=float)
img_vector_len = [[np.sqrt(np.sum(np.square(params[i, j, :]))) for j in range(img_width)] for i in range(img_height)]
# norm_params2 = np.array([[[(img[k][i][j] / img_vector_len[k][i]) if img_vector_len[k][i] != 0 else 0 for j in
#                           range(channel_count)] for i in range(img_width)] for k in range(img_height)])
norm_params = [[params[i, j, :] / img_vector_len[i][j] if img_vector_len[i][j] != 0 else 0 for j in range(img_width)] for i in range(img_height)]
# for i in range(1):
#     print((((img[0][0][0] ** 2) + (img[0][0][1] ** 2) + (img[0][0][2] ** 2)) ** 0.5))
#     print(np.sqrt(np.square(img2[0][0][:])))
#     print(img[0][0][:])
#     print(img_vector_len[i], "\n", img_vector_lena[i])
# print(np.array_equal(img_vector_lena, img_vector_len))
# print(len(img_vector_lena), "\n", len(img_vector_len))
# print(end_time_1 - start_time)
# print(end_time_2 - start_time2)

params = np.array(img_fragment, dtype=float)
img_fragment_vector_len = [[np.sqrt(np.sum(np.square(params[i, j, :]))) for j in range(img_fragment_width)] for i in range(img_fragment_height)]
norm_fragment_params = [[params[i, j, :] / img_fragment_vector_len[i][j] if img_fragment_vector_len[i][j] != 0 else 0 for j in range(img_fragment_width)] for i in range(img_fragment_height)]

# img_fragment_vector_len = [
#     [(((img_fragment[i][j][0] ** 2) + (img_fragment[i][j][1] ** 2) + (img_fragment[i][j][2] ** 2)) ** 0.5) for j in
#      range(img_fragment_width)] for i in range(img_fragment_height)]
# norm_fragment_params = np.array([[[(img_fragment[k][i][j] / img_fragment_vector_len[k][i]) if
#                                    img_fragment_vector_len[k][i] != 0 else 0 for j in range(channel_count)] for i in
#                                   range(img_fragment_width)] for k in range(img_fragment_height)])

# Задаются 24 веса для 3 входных каналов RGB и 8 выходов
weights = np.random.random((channel_count, cluster_count))
weights_vector_len = [(((weights[0][i] ** 2) + (weights[1][i] ** 2) + (weights[2][i] ** 2)) ** 0.5) for i in
                      range(cluster_count)]
norm_weights = np.array(
    [[weights[i][j] / weights_vector_len[j] for j in range(cluster_count)] for i in range(channel_count)])

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

clusters = np.zeros((img_fragment_height, img_fragment_width), dtype=int)
for k in range(img_fragment_height):
    for i in range(img_fragment_width):
        clusters[k][i] = np.argmax(np.dot(norm_fragment_params[k][i], norm_weights))

resolution = img_fragment_height * img_fragment_width
clusters = np.unique(clusters, return_counts=True)
clusters = np.around(clusters[1] / resolution, 2)


buffer = np.zeros((img_height, img_width), dtype=int)
for k in range(img_height):
    for i in range(img_width):
        buffer[k][i] = np.argmax(np.dot(norm_params[k][i], norm_weights))

for k in range(img_height - img_fragment_height):
    os.system('cls||clear')
    print("Поиск фрагмента завершён на ", (k+1)/(img_height-img_fragment_height) * 100, "%")
    for i in range(img_width - img_fragment_width):
        temp = np.unique(buffer[k:k + img_fragment_height, i:i + img_fragment_width], return_counts=True)
        temp = np.around(temp[1] / resolution, 2)
        if np.array_equal(clusters, temp):
            cv2.rectangle(img, (i, k), (i + img_fragment_width, k + img_fragment_height), (0, 0, 255), 1)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

