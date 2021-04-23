import subprocess
import os
import cv2

if __name__ == '__main__':
    svbrdf_dir = './bin/test8'
    rgb_albedo_filename = os.path.join(svbrdf_dir, 'albedo_rgb.png')
    lab_albedo_filename = os.path.join(svbrdf_dir, 'albedo.png')
    mask_filename = os.path.join(svbrdf_dir, 'mask.png')
    # convert rgb to lab
    src = cv2.imread(rgb_albedo_filename)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    cv2.imwrite(lab_albedo_filename, src)
    # patchmatch
    subprocess.run('./bin/ebsynth.exe -svbrdf_dir {} -mask {} -searchvoteiters 100 -patchmatchiters 100 -extrapass3x3'.format(svbrdf_dir, mask_filename))
    # post processing
    inpainted_filename = os.path.join(svbrdf_dir, 'albedo_inpainted.png')
    src = cv2.imread(inpainted_filename)
    src = cv2.cvtColor(src, cv2.COLOR_Lab2BGR)
    cv2.imwrite(inpainted_filename, src)
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape[:2]
    print(width, height)
    minx = width
    maxx = 0
    miny = height
    maxy = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 255:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    # img = cv2.rectangle(mask, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
    # cv2.imshow('test', mask)
    # cv2.waitKey(0)
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    inpaint_radius = int(0.01 * max(width, height))
    dst = cv2.inpaint(src, mask.copy(), inpaint_radius, cv2.INPAINT_TELEA)
    opencv_inpainted_filename = os.path.join(svbrdf_dir, 'opencv_inpainted.png')
    cv2.imwrite(opencv_inpainted_filename, dst)

    center = (minx + (maxx - minx) // 2, miny + (maxy - miny) // 2)

    print(center)

    clone = cv2.seamlessClone(src, dst, mask, center, cv2.MONOCHROME_TRANSFER)
    opencv_clone_filename = os.path.join(svbrdf_dir, 'opencv_clone.png')
    cv2.imwrite(opencv_clone_filename, clone)
