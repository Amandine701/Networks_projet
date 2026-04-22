import geopandas as gpd

# Il se peut qu'un time out arrive lors du téléchargement, il suffit de relancer le code
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_JPN_2.json"
japan = gpd.read_file(url).to_crs(3857)
japan.to_file("data/japan.geo.json", driver="GeoJSON")

tokyo_large = japan[japan["NAME_1"] == "Tokyo"]
tokyo_large.to_file("data/tokyo.geo.json", driver="GeoJSON")

tokyo_wards = tokyo_large[tokyo_large["TYPE_2"] == "SpecialWard"]
tokyo_wards.to_file("data/tokyo_wards.geo.json", driver="GeoJSON")

# Les arrondissements les plus denses de Tokyo
central_wards = ["Shibuya", "Shinjuku", "Chiyoda", "Chūō", "Minato", "Taitō",
                 "Bunkyō", "Sumida", "Nakano", "Toshima", "Meguro", "Shinagawa"]

tokyo_center = tokyo_wards[tokyo_wards["NAME_2"].isin(central_wards)]
tokyo_center.to_file("data/tokyo_center.geo.json", driver="GeoJSON")