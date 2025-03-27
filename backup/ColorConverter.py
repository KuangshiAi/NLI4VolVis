import numpy as np

class ColorSpaceConverter():
    """
    To avoid rounding point errors, this module implements the RGB to HSV conversion algorithm as presented in
    "Integer-based accurate conversion between RGB and HSV color spaces"
    by Vladimir Chernov, Jarmo Alander, Vladimir Bochko

    Source: https://bitbucket.org/ColorSpaceConverter/colormath_hsv/src
    """

    E = int(65537)
    S_MAX = int(65535)
    H_MAX = int(393222)

    def rgb_to_hsv(self, r: int, g: int, b: int) -> tuple[ int, int, int]:
        """Convert RGB to an integer representation of HSV."""
        k=[r,g,b]
        k.sort()
        mn, mid, mx = k

        v = mx
        d = mx - mn

        if d == 0:
            return 0, 0, v
        
        if mx == r and mn == b:
            i=0
        elif mx == g and mn == b:
            i=1
        elif mx == g and mn == r:
            i=2
        elif mx == b and mn == r:
            i=3
        elif mx == b and mn == g:
            i=4
        else:
            i=5

        s = ((d << 16) - 1) // v

        f: int = (((mid - mn) << 16) // d) + 1

        if i == 1 or i == 3 or i == 5:
            f = ColorSpaceConverter.E - f

        h = (ColorSpaceConverter.E * i) + f
        return h, s, v

    def hsv_to_rgb(self, h: int, s: int, v: int):
        """Convert integer representation of HSV to RGB."""
        if s == 0 or v == 0:
            return v, v, v
        
        d = ((s * v) >> 16) + 1
        mn = v - d

        if h < ColorSpaceConverter.E:
            i=0
        elif h >= ColorSpaceConverter.E and h < 2 * ColorSpaceConverter.E:
            i=1
        elif h >= 2 * ColorSpaceConverter.E and h < 3 * ColorSpaceConverter.E:
            i=2
        elif h >= 3 * ColorSpaceConverter.E and h < 4 * ColorSpaceConverter.E:
            i=3
        elif h >= 4 * ColorSpaceConverter.E and h < 5 * ColorSpaceConverter.E:
            i=4
        else:
            i=5

        if i == 1 or i == 3 or i == 5:
            f = ColorSpaceConverter.E * (i + 1) - h
        else:
            f = h - (ColorSpaceConverter.E * i)

        mid = ((d * f) >> 16) + mn

        if i == 0:
            return v, mid, mn
        elif i == 1:
            return mid, v, mn
        elif i == 2:
            return mn, v, mid
        elif i == 3:
            return mn, mid, v
        elif i == 4:
            return mid, mn, v
        else:
            return v, mn, mid

    def rgb_to_hsv_degrees(self, r: int, g: int, b: int) -> tuple[ int, int, int]:
        """Convert RGB to an integer representation of HSV."""
        k=[r,g,b]
        k.sort()
        mn, mid, mx = k

        v = mx
        d = mx - mn

        if d == 0:
            return 0, 0, v * 100 // 255
        
        if mx == r and mn == b:
            i=0
        elif mx == g and mn == b:
            i=1
        elif mx == g and mn == r:
            i=2
        elif mx == b and mn == r:
            i=3
        elif mx == b and mn == g:
            i=4
        else:
            i=5

        s = ((d << 16) - 1) // v

        f: int = (((mid - mn) << 16) // d) + 1

        if i == 1 or i == 3 or i == 5:
            f = ColorSpaceConverter.E - f

        h = (ColorSpaceConverter.E * i) + f
        return h * 360 // ColorSpaceConverter.H_MAX, s * 100 // ColorSpaceConverter.S_MAX , v * 100 // 255


if __name__ == "__main__":
    o = ColorSpaceConverter()
    rgb = [255, 15, 0]
    hsv = ColorSpaceConverter().rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    rgb = ColorSpaceConverter().hsv_to_rgb(hsv[0], hsv[1], hsv[2])
    print(hsv)
    print(rgb)