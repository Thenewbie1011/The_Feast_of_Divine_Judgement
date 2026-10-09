from database import foods_collection
foods=[
    {
        "class_name":"Bread",
        "description":"A baked food made primarily from flour and water. NOTE: Nutritional information taken from USDA",
        "calories":266,
        "protein":8.85,
        "carbohydrates":49.4,
        "link":"https://fdc.nal.usda.gov/food-details/174924/nutrients"
    },
    {
        "class_name":"Egg",
        "description":"A nutrient-rich food commonly eaten boiled, fried, scrambled, or as part of other dishes. NOTE: Nutritional information"
        "taken from USDA",
        "calories":155,
        "protein":12.6,
        "carbohydrates":1.12,
        "link":"https://fdc.nal.usda.gov/food-details/173424/nutrients"
    },
    {
        "class_name":"Rice",
        "description":"A staple grain commonly eaten as a main food or as part of other dishes. NOTE: Nutritional information taken from USDA",
        "calories":358,
        "protein":6.69,
        "carbohydrates":80.1,
        "link":"https://fdc.nal.usda.gov/food-details/1104867/nutrients"
    },
    {
        "class_name":"Fried Food",
        "description":"Food cooked by immersing or cooking in hot oil. NOTE: Nutritional information taken from USDA",
        "calories":317,
        "protein":18.12,
        "carbohydrates":2.15,
        "link":"https://fdc.nal.usda.gov/food-details/746780/nutrients"
    },
    {
        "class_name":"Dessert",
        "description":"Sweet foods commonly served at the end of a meal. NOTE: Nutritional information taken from USDA",
        "calories":383,
        "protein":0,
        "carbohydrates":99,
        "link":"https://fdc.nal.usda.gov/food-details/168792/nutrients"
    },
    {
        "class_name":"Dairy products",
        "description":"Foods made from milk, such as cheese, yogurt, and other milk-based products. NOTE: Nutritional information taken from USDA",
        "calories":197,
        "protein":10.3,
        "carbohydrates":5.48,
        "link":"https://fdc.nal.usda.gov/food-details/172200/nutrients"
    },
    {
        "class_name":"Noodles/Pasta",
        "description":"Foods made primarily from flour and water, commonly served with sauces or other ingredients. NOTE: Nutritional information taken"
        "from USDA",
        "calories":108,
        "protein":1.79,
        "carbohydrates":24,
        "link":"https://fdc.nal.usda.gov/food-details/168914/nutrients"
    },
    {
        "class_name":"Meat",
        "description":"Animal-based food that is a major source of protein and other nutrients. NOTE: Nutritional information taken form USDA",
        "calories":113,
        "protein":18.7,
        "carbohydrates":0,
        "link":"https://fdc.nal.usda.gov/food-details/173638/nutrients"
    },
    {
        "class_name":"Seafood",
        "description":"Edible aquatic animals such as fish, shrimp, and other marine foods. NOTE: Nutritional information taken from USDA",
        "calories":190,
        "protein":10.12,
        "carbohydrates":1.25,
        "link":"https://fdc.nal.usda.gov/food-details/2706838/nutrients"
    },
    {
        "class_name":"Soup",
        "description":"A liquid-based dish prepared with ingredients such as vegetables, meat, or seafood. NOTE: Nutritional information taken from USDA",
        "calories":37,
        "protein":2.53,
        "carbohydrates":5.71,
        "link":"https://fdc.nal.usda.gov/food-details/172899/nutrients"
    },
    {
        "class_name":"Vegetable/Fruit",
        "description":"Plant-based foods that provide vitamins, minerals, fiber, and other nutrients. NOTE: Nutritional information taken from USDA",
        "calories":65,
        "protein":0.15,
        "carbohydrates":15.6,
        "link":"https://fdc.nal.usda.gov/food-details/1750340/nutrients"
    }
]
result=foods_collection.insert_many(foods)
print(f"Inserted {len(result.inserted_ids)} food documents.")