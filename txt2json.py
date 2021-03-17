import json
data = []
with open("model/Decoder/Generated_Captions_MSCOCO_154.txt", 'r') as caption_file:
    for line in caption_file:
        image_id = line[:13]
        caption = line[13:]

        data.append({"image_id": image_id, "caption": caption})
    json_file = "model/Decoder/Generated_Captions_MSCOCO_154.json"
    with open(json_file, "a") as file:
        json.dump(data, file)
print("done")