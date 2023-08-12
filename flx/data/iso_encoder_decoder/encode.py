from math import *

# convert minutiae template from txt format to iso19794-2-2005 format.


def to_iso19794(txtpath, isopath):
    width, height = 500, 610
    b_array = bytearray()
    with open(txtpath, "r") as textfile:
        lines = textfile.readlines()
    minutiae_num = len(lines)
    totalbytes = minutiae_num * 6 + 28 + 2

    b_array = bytearray(b"FMR\x00 20\x00")
    b_array += totalbytes.to_bytes(4, "big")
    b_array += bytearray(b"\x00\x00")
    b_array += width.to_bytes(2, "big")
    b_array += height.to_bytes(2, "big")
    b_array += bytearray(b"\x00\xc5\x00\xc5\x01\x00\x00\x00d")
    b_array += minutiae_num.to_bytes(1, "big")
    byte_list = [0, 0, 0, 0, 0, 0]
    for line in lines:
        line = line.split(" ")
        x, y, angle, min_type, quality = (
            int(float(line[0])),
            int(float(line[1])),
            int(float(line[2])),
            int(float(line[3])),
            int(float(line[4].strip())),
        )
        byte_list[1] = x % 256
        byte_list[0] = x // 256 + min_type * 64
        byte_list[2] = y // 256
        byte_list[3] = y % 256
        byte_list[4] = round((360 - angle) / 360 * 256)
        byte_list[5] = quality
        b_array += bytearray(byte_list)

    zero = 0
    b_array += zero.to_bytes(2, "big")

    with open(isopath, "wb") as istfile:
        istfile.write(b_array)


def main():
    txtpath = "txtTemplate.txt"
    isopath = "isoTemplate"
    to_iso19794(txtpath, isopath)


if __name__ == "__main__":
    main()
