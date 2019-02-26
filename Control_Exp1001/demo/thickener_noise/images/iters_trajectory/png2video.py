#!/
import os
import collections
import cv2
import glob
import shutil
import time

#创建若干个文件夹
def mk_dir():
    for i in range(1,10):
        file_name = 'pic-'+str(i)
        if os.path.exists(file_name):
            os.removedirs()
        os.mkdir(file_name)


#将同一视频文件的图片放到同一文件夹下
def classify():
    path = './sum/'
    pwd = os.getcwd()
    for files in os.listdir(path):
        #print(files)
        num = files.split('-')[0]
        if not num.isalnum():
            continue

        dest_dir_path = os.path.join(pwd, 'pic-'+num)
        if not os.path.exists(dest_dir_path):
            os.mkdir(dest_dir_path)
        if os.path.exists(os.path.join(dest_dir_path, files)):
            os.remove(os.path.join(dest_dir_path, files))
        shutil.copy(path + files, dest_dir_path)
        #print(num)


#将文件夹下的图片合成视频，存储在当前文件夹下
def transfer(path):
    print(path)
    # path = r'C:\Users\Administrator\Desktop\1\huaixiao\\'#文件路径
    filelist = os.listdir(path)  # 获取该目录下的所有文件名
    # print(filelist)
    # filelist.sort(key=lambda x: int(x[:-4].split('-')[0]))

    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 6
    size = (640,480) #图片的分辨率片
    file_path = path +'-'+ str(int(time.time())) + ".mp4"  # 导出路径
    print(file_path)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(file_path, fourcc, fps, size)

    # for item in filelist:
    #     # print(item)
    #     # print(item.split('-')[1][:-4])
    #     filelist.sort(key=lambda x: int(item.split('-')[1][:-4]))
    filelist.sort(key=lambda x: int(x.split('-')[1][:-4]))

    for item in filelist:
        print(item)
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频

    video.release()  # 释放


if __name__ == '__main__':
    pwd = os.getcwd()
    #print(pwd)
    classify()
    for fileName in os.listdir(pwd):
        #print(fileName)
        if fileName.split('-')[0] == 'pic' and os.path.isdir(fileName) and len(os.listdir(fileName)) > 0:
            print(pwd+'/'+fileName)
            transfer(pwd+'/'+fileName)







