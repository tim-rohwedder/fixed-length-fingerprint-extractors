from collections import namedtuple

# This code is to decode the iso19794-2-2005 and iso-19794-2-2011 templates of fingerprint minutiae.
# VeriFinger SDK could recognize both of these two formats.

Minutiae = namedtuple(
    "Minutiae",
    ["type", "x", "y", "orientation", "quality", "label"],
    defaults=[None, None],
)


def load_iso19794(path, format):
    if format == "19794-2-2005":
        with open(path, "rb") as f:
            t = f.read()
        magic = int.from_bytes(t[0:4], "big")
        version = int.from_bytes(t[4:8], "big")
        total_bytes = int.from_bytes(t[8:12], "big")
        im_w = int.from_bytes(t[14:16], "big")
        im_h = int.from_bytes(t[16:18], "big")
        resolution_x = int.from_bytes(t[18:20], "big")
        resolution_y = int.from_bytes(t[20:22], "big")
        f_count = int.from_bytes(t[22:23], "big")
        reserved_byte = int.from_bytes(t[23:24], "big")
        fingerprint_quality = int.from_bytes(t[26:27], "big")
        minutiae_num = int.from_bytes(t[27:28], "big")
        minutiaes = []
        for i in range(minutiae_num):
            x = 28 + 6 * i
            min_type = (t[x] >> 6) & 0x3
            min_x = int.from_bytes([t[x] & 0x3F, t[x + 1]], "big")
            min_y = int.from_bytes(t[x + 2 : x + 4], "big")
            angle = 360 - t[x + 4] / 256 * 360
            min_quality = t[x + 5]
            minutiaes.append(Minutiae(min_type, min_x, min_y, angle, min_quality))
        return minutiaes

    if format == "19794-2-2011":
        with open(path, "rb") as f:
            t = f.read()
        magic = int.from_bytes(t[0:4], "big")
        version = int.from_bytes(t[4:8], "big")
        total_bytes = int.from_bytes(t[8:12], "big")
        fp_count = int.from_bytes(t[12:14], "big")
        HASCERTS = int.from_bytes(t[14:15], "big")
        fp_bytes = int.from_bytes(t[15:19], "big")
        year = int.from_bytes(t[19:21], "big")
        month = int.from_bytes(t[21:22], "big")
        day = int.from_bytes(t[22:23], "big")
        hour = int.from_bytes(t[23:24], "big")
        minute = int.from_bytes(t[24:25], "big")
        second = int.from_bytes(t[25:26], "big")
        millisecond = int.from_bytes(t[26:28], "big")
        dev_tech = int.from_bytes(t[28:29], "big")
        dev_vendor = int.from_bytes(t[29:31], "big")
        dev_id = int.from_bytes(t[31:33], "big")
        QCOUNT = int.from_bytes(t[33:34], "big")
        FP_quality = int.from_bytes(t[34:35], "big")
        Q_vendor = int.from_bytes(t[35:37], "big")
        Q_algo = int.from_bytes(t[37:39], "big")
        position = int.from_bytes(t[39:40], "big")
        view_offset = int.from_bytes(t[40:41], "big")
        resolution_x = int.from_bytes(t[41:43], "big")
        resolution_y = int.from_bytes(t[43:45], "big")
        sample_type = int.from_bytes(t[45:46], "big")
        im_w = int.from_bytes(t[46:48], "big")
        im_h = int.from_bytes(t[48:50], "big")
        MINBYTES, ENDINGTYPE = (t[50] >> 4) & 0xF, t[50] & 0xF
        minutiae_num = int.from_bytes(t[51:52], "big")
        minutiaes = []
        for i in range(minutiae_num):
            x = 52 + 6 * i
            min_type = (t[x] >> 6) & 0x3
            min_x = int.from_bytes([t[x] & 0x3F, t[x + 1]], "big")
            min_y = int.from_bytes(t[x + 2 : x + 4], "big")
            angle = 360 - t[x + 4] / 256 * 360
            min_quality = t[x + 5]
            minutiaes.append(Minutiae(min_type, min_x, min_y, angle, min_quality))
        return minutiaes


def main():
    minutiae_2005 = load_iso19794("iso2005template", format="19794-2-2005")
    print(minutiae_2005)

    minutiae_2011 = load_iso19794("iso2011template", format="19794-2-2011")
    print(minutiae_2011)


if __name__ == "__main__":
    main()
