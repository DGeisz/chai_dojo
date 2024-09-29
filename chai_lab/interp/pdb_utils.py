def pdbid_to_int(pdb_id: str):
    return int(pdb_id.upper(), 36)


def int_to_pdbid(number):
    base36_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower()
    result = []
    while number > 0:
        result.append(base36_chars[number % 36])
        number //= 36
    return "".join(reversed(result)).zfill(4)  # Ensure it is 4 characters long
