import cv2
import numpy as np
from matplotlib import pyplot as plt


def RGBtoCMYK():
    # Import image
    img = plt.imread("rgb.jpg")
    # Create float
    bgr = img.astype(float) / 255.
    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = (1 - bgr[..., 2] - K) / (1 - K)
        M = (1 - bgr[..., 1] - K) / (1 - K)
        Y = (1 - bgr[..., 0] - K) / (1 - K)
    # Convert the input BGR image to CMYK colorspace
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    # View image
    CMYK = cv2.resize(CMYK, (0, 0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('RGB', img)
    cv2.imshow('CMYK', CMYK)
    cv2.imwrite('CMYK.jpg', CMYK)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def CMYKtoRGB():
    # Import image
    img = plt.imread("CMYK.jpg")
    # Create float
    cmyk = img.astype(float) / 100
    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = (1 - np.min(cmyk, axis=2)) / 100
        R = 255 * (1 - cmyk[..., 2]) * (1 - K)
        G = 255 * (1 - cmyk[..., 1]) * (1 - K)
        B = 255 * (1 - cmyk[..., 0]) * (1 - K)
    # Convert the input BGR image to CMYK colorspace
    RGB = (np.dstack((B, G, R)) * 100).astype(np.uint8)
    # View image
    RGB = cv2.resize(RGB, (0, 0), fx=1.0, fy=1.0)
    img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
    cv2.imshow('CMYK', img)
    cv2.imshow('RGB', RGB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RGBtoHSV():
    img = cv2.imread('rgb.jpg')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # View image
    img_hsv = cv2.resize(img_hsv, (0, 0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('RGB', img)
    cv2.imshow('HSV', img_hsv)
    cv2.imwrite('HSV.jpg', img_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HSVtoRGB():
    img = cv2.imread('HSV.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # View image
    img_rgb = cv2.resize(img_rgb, (0, 0), fx=1.0, fy=1.0)
    img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
    cv2.imshow('HSV', img)
    cv2.imshow('RGB', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RGBtoHSL():
    img = cv2.imread('rgb.jpg')
    img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # View image
    img_hsl = cv2.resize(img_hsl, (0, 0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('RGB', img)
    cv2.imshow('HSL', img_hsl)
    cv2.imwrite('HSL.jpg', img_hsl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HSLtoRGB():
    img = cv2.imread('HSL.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    # View image
    img_rgb = cv2.resize(img_rgb, (0, 0), fx=1.0, fy=1.0)
    img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
    cv2.imshow('HSL', img)
    cv2.imshow('RGB', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RGBtoYCBCR():
    img = cv2.imread('rgb.jpg')
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # View image
    img_ycbcr = cv2.resize(img_ycbcr, (0, 0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('RGB', img)
    cv2.imshow('YCBCR', img_ycbcr)
    cv2.imwrite('YCBCR.jpg', img_ycbcr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def YCBCRtoRGB():
    img = cv2.imread('YCBCR.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    # View image
    img_rgb = cv2.resize(img_rgb, (0, 0), fx=1.0, fy=1.0)
    img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
    cv2.imshow('YCBCR', img)
    cv2.imshow('RGB', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    while 1:
        print("1. RGB -> CMYK")
        print("2. RGB -> HSV")
        print("3. RGB -> HSL")
        print("4. RGB -> YCBCR")
        print("5. CMYK -> RGB")
        print("6. HSV -> RGB")
        print("7. HSL -> RGB")
        print("8. YCBCR -> RGB")
        print("9. Exit Program")
        print("Choose Option : ")
        pil = int(input())
        if pil == 1:
            RGBtoCMYK()
        elif pil == 2:
            RGBtoHSV()
        elif pil == 3:
            RGBtoHSL()
        elif pil == 4:
            RGBtoYCBCR()
        elif pil == 5:
            CMYKtoRGB()
        elif pil == 6:
            HSVtoRGB()
        elif pil == 7:
            HSLtoRGB()
        elif pil == 8:
            YCBCRtoRGB()
        elif pil == 9:
            exit()
        else:
            print("Invalid Option")


main()
