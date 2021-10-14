from data.image.image_entity import Image


if __name__ == '__main__':
    image = Image(frame_index=0)
    image.cv2_resize()
    image.cv2_toNumpy()
    image.normalize()
