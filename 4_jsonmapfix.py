import json

inputfilepath = "filepath"
outputfilepath = "filepath"

# Get json shape file
with open(inputfilepath, "r") as file:
    data = json.load(file)
    for row in data["objects"]["CITYPLAN_CRA_-8527542012581552321"]["geometries"]:
        row["properties"]["CRA_NO"] = round(row["properties"]["CRA_NO"], 1)
        print(row["properties"]["CRA_NO"])

# Filter out water shapes
keys_to_delete = [
    geometry for geometry in data["objects"]["CITYPLAN_CRA_-8527542012581552321"]["geometries"]
    if geometry["properties"]["WATER"] == 1
]
filtered_geometries = [
    geometry for geometry in data["objects"]["CITYPLAN_CRA_-8527542012581552321"]["geometries"]
    if geometry not in keys_to_delete
]
data["objects"]["CITYPLAN_CRA_-8527542012581552321"]["geometries"] = filtered_geometries

# Write new json
with open(outputfilepath, "w") as file2:
    json.dump(data, file2)
