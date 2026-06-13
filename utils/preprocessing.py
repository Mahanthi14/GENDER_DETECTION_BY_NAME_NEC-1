def extract_features(name):

    name = name.lower()

    return {
        "first_letter": name[0],
        "last_letter": name[-1],
        "name_length": len(name)
    }