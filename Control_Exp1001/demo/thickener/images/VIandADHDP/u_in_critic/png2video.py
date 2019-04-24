#!/
import os
import collections
import cv2
import glob
import shutil
import time



#创建若干个文件夹

def merge_images(img_name_list, video_path):


    fps = 2
    size = (640,480) #图片的分辨率片
    print('num of img: ', len(img_name_list))
    print(video_path)
    #fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv2.VideoWriter(video_path, fourcc, fps, size)
    for item in img_name_list:
        if item.endswith('.png'):  # 判断图片后缀是否是.png

            print(item)
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            status = video.write(img)  # 把图片写进视频
            print(status)
    video.release()  # 释放


def scan_img_list(root_path, sort_rule=None):
    if sort_rule is None:
        raise ValueError
    img_name_list = []
    for file in os.listdir(root_path):
        if file.endswith('png'):
            img_name_list.append(file)
    img_name_list.sort(key=sort_rule)
    img_name_list = list(map(lambda x:os.path.join(root_path, x), img_name_list))
    return img_name_list

if __name__ == '__main__':
    root_path = os.getcwd()
    sort_rule = lambda x:int(x.split('-')[1][:-4])
    img_name_list = scan_img_list(root_path, sort_rule=sort_rule)
    video_path = os.path.join(root_path, 'action_trajectory.mp4')
    merge_images(img_name_list, video_path)
