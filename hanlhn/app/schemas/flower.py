
def flowerEntity(item) -> dict:
    return {
        "amount": str(item["num"]),
        "eng_name": str(item["English name"]),
        "vie_name": str(item["Vietnam name"]),
        "description": str(item["Description"])
    }

def flowersEntity(entity) -> list:
    return [flowerEntity(item) for item in entity]