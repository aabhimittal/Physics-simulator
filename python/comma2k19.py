"""Fetches drives of a 2016 Honda Civic from comma2k19 for tests/vehicle.rs.

    python3 python/comma2k19.py [chunk] [drives] > tests/data/comma2k19.csv    # defaults: 5 12

comma2k19 (comma.ai, MIT licence, https://github.com/commaai/comma2k19) is 33 hours of
highway commuting in one-minute segments, about 100 GB in all. Its Hugging Face mirror
stores each chunk as one zip, and a zip's index sits at its end, so this reads just the
index and the few small arrays it needs over HTTP range requests (about 0.5 MB a drive)
instead of downloading a 10 GB chunk. Chunks 3 to 10 are the Civic.

Each output row: drive number, time (s), speed (m/s, from CAN), steering-wheel angle
(degrees, positive left, from CAN) and yaw rate (rad/s, positive left, from the phone IMU's
gyro, smoothed over 0.1 s), at 10 Hz.
"""

import io
import sys
import urllib.request
import zipfile

import numpy as np

URL = "https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{}.zip"


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size, self.url = int(r.headers["Content-Length"]), r.url
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        wanted = {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=wanted)) as r:
            data = r.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def array(z, segment, name):
    return np.load(io.BytesIO(z.read(f"{segment}/processed_log/{name}")))


def main():
    chunk = sys.argv[1] if len(sys.argv) > 1 else "5"
    drives = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    z = zipfile.ZipFile(io.BufferedReader(RemoteFile(URL.format(chunk)), 1 << 16))
    segments = sorted({n.split("/processed_log/")[0] for n in z.namelist() if "/processed_log/" in n})
    print("# comma2k19 (Copyright (c) 2018 comma.ai, MIT licence), chunk " + chunk + ", via python/comma2k19.py")
    print("# drive,time_s,speed_mps,steering_wheel_deg,yaw_rate_radps")
    for k, segment in enumerate(segments[:: len(segments) // drives][:drives]):
        speed_t, speed = array(z, segment, "CAN/speed/t"), array(z, segment, "CAN/speed/value").ravel()
        steer_t, steer = array(z, segment, "CAN/steering_angle/t"), array(z, segment, "CAN/steering_angle/value")
        gyro_t, gyro = array(z, segment, "IMU/gyro/t"), array(z, segment, "IMU/gyro/value")
        start, end = max(speed_t[0], steer_t[0], gyro_t[0]), min(speed_t[-1], steer_t[-1], gyro_t[-1])
        t = np.arange(start, end, 0.1)
        # The IMU's axes are forward, right, down: yaw to the left is minus the down axis.
        yaw = -np.convolve(gyro[:, 2], np.ones(10) / 10, mode="same")
        for row in zip(t - start, np.interp(t, speed_t, speed), np.interp(t, steer_t, steer.ravel()), np.interp(t, gyro_t, yaw)):
            print(f"{k},{row[0]:.1f},{row[1]:.3f},{row[2]:.2f},{row[3]:.5f}")
        print(f"fetched {segment}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
