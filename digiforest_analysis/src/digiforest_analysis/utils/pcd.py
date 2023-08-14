def load_header(file, binary=True):
    header = {}
    open_mode = "rb" if binary else "r"
    with open(file, open_mode) as f:
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


def replace_file_header(file, header, binary=True):
    open_mode = "wb+" if binary else "w+"
    with open(file, open_mode) as f:
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

        file_content = f.read()

        for key, value in header.items():
            header_entry = key
            if isinstance(value, str):
                header_entry += f" {value}"
            elif isinstance(value, list):
                for v in value:
                    header_entry += f" {v}"
            header_entry = header_entry.encode()
            file_content.replace(header_entry)
