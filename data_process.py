import cv2
import numpy as np


def count_cell(image,ori_image):
    gray = image
    gray1 = cv2.GaussianBlur(gray, (3, 3), 0)
    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    gray2 = gray1
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, kernel)
    binary_image = erosion
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity_threshold = 0.2
    color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    results = []
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    custom_color_map = {
        1: (255, 0, 0),  # 红色
        2: (0, 255, 0),  # 绿色
        3: (0, 0, 255),  # 蓝色
    }
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > circularity_threshold:
            coordinates = [tuple(point[0]) for point in contour]
            results.append({
                'index': i + 1,
                'circularity': circularity,
                'area': area,
                'coordinates': coordinates
            })
            if (i + 1) in custom_color_map:
                color = custom_color_map[i + 1]
            else:
                color_index = (i // len(color_map)) % len(color_map)
                color = color_map[color_index]

            cv2.drawContours(color_image, [contour], -1, color, thickness=-1)
            cv2.putText(color_image, str(i + 1), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    count = len(results)
    height, width, _ = color_image.shape
    font_scale = 3.0
    text = f"CELL COUNT = {count}"
    text_color = (0, 255, 0)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_x = 10
    text_y = 10 + text_size[1]
    cv2.putText(color_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
    result_string = ""
    result_string += f"图片菌落总数: {count} \n"
    # result_string += "符合要求的菌落信息：\n"
    for result in results:
        center_x = sum(point[0] for point in result['coordinates']) / len(result['coordinates'])  # 连通块中心 x 坐标
        center_y = sum(point[1] for point in result['coordinates']) / len(result['coordinates'])  # 连通块中心 y 坐标
        result_string += f"编号：{result['index']}，圆形度：{result['circularity']:.2f}，面积：{result['area']:.2f}，中心坐标信息：x={center_x:.2f}，y={center_y:.2f}\n"

    merge_image = merge(ori_image, color_image)
    return result_string, merge_image


def merge(original_image,label_image ):
    alpha = 0.5
    beta = 1.0 - alpha
    original_image=np.array(original_image)
    overlay_image = cv2.addWeighted(original_image, alpha, label_image, beta, 0)
    return overlay_image


