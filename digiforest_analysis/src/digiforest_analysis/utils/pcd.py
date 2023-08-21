import fileinput
import open3d as o3d
import sys


def load(filename: str, binary=True):
    cloud = o3d.t.io.read_point_cloud(filename)
    header = load_header(filename, binary=binary)
    return cloud, header


def write(cloud, header, filename):
    write_open3d(cloud, header, filename)


def write_pcl(cloud, header, filename):
    # Write cloud to file
    cloud.to_file(str.encode(filename), ascii=False)

    # Update header
    original_header = load_header(filename, binary=True)
    for k, v in header.items():
        original_header[k] = v

    # Update file with header
    replace_file_header(filename, original_header, binary=True)


def write_open3d(cloud, header, filename):
    # Write cloud to file
    o3d.io.write_point_cloud(filename, cloud.to_legacy())

    # Update header
    original_header = load_header(filename, binary=True)
    for k, v in header.items():
        original_header[k] = v

    # Update file with header
    replace_file_header(filename, original_header, binary=True)


def load_header(filename, binary=True):
    header = {}
    open_mode = "rb" if binary else "r"
    with open(filename, open_mode) as f:
        while True:
            line = f.readline().decode()

            if line.startswith("#"):
                continue
            if line.startswith("VERSION"):
                key, value = line.split()
                header[key] = value
            elif (
                line.startswith("FIELDS")
                or line.startswith("SIZE")
                or line.startswith("TYPE")
                or line.startswith("COUNT")
                or line.startswith("VIEWPOINT")
            ):
                parts = line.split()
                key = parts[0]
                values = parts[1:]
                header[key] = values
            elif (
                line.startswith("WIDTH")
                or line.startswith("HEIGHT")
                or line.startswith("POINTS")
            ):
                key, value = line.split()
                header[key] = int(value)
            elif line.startswith("DATA"):
                # We ignore the data fields
                break
    return header


def replace_file_header(filename, header, binary=True):
    if binary:
        replace_file_header_binary(filename, header)
    else:
        replace_file_header_normal(filename, header)


def replace_file_header_normal(filename, header):
    pass


def replace_file_header_binary(filename, header):
    encoding = "utf-8"
    with fileinput.FileInput(filename, inplace=True, mode="rb") as f:
        for line in f:
            # Parse file
            if line.startswith(b"#"):
                pass
            if line.startswith(b"VERSION"):
                key, value = line.decode().split()
                # line.replace(line, (f"{key} {header[key]}\n").encode(encoding))
                line = (f"{key} {header[key]}\n").encode(encoding)
            elif (
                line.startswith(b"FIELDS")
                or line.startswith(b"SIZE")
                or line.startswith(b"TYPE")
                or line.startswith(b"COUNT")
                or line.startswith(b"VIEWPOINT")
            ):
                parts = line.decode().split()
                key = parts[0]
                # line.replace(line, (f"{key} " + " ".join(header[key]) + "\n").encode(encoding))
                line = (f"{key} " + " ".join(header[key]) + "\n").encode(encoding)
            elif (
                line.startswith(b"WIDTH")
                or line.startswith(b"HEIGHT")
                or line.startswith(b"POINTS")
            ):
                key, value = line.decode().split()
                # line.replace(line, (f"{key} {header[key]}\n").encode(encoding))
                line = (f"{key} {header[key]}\n").encode(encoding)
            elif line.startswith(b"DATA"):
                pass
            else:
                pass

            # Write to standard output
            sys.stdout.write(line)
