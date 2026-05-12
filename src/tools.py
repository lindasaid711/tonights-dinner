import json
import requests
from typing import List
from langchain_core.tools import tool  # noqa: F401 (re-exported)

_recipe_cache: dict = {}


def _fetch_from_mealdb(query: str) -> list:
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    if not data.get("meals"):
        return []

    results = []
    for meal in data["meals"][:10]:
        ingredients = []
        for i in range(1, 21):
            ing = (meal.get(f"strIngredient{i}") or "").strip()
            if ing:
                ingredients.append(ing)
        results.append({
            "name": meal["strMeal"],
            "cuisine": meal.get("strArea", "Unknown"),
            "category": meal.get("strCategory", ""),
            "ingredients": ingredients,
        })
    return results


@tool
def search_recipes(query: str) -> str:
    """Search TheMealDB for dinner recipes matching a query.
    Returns a JSON array of recipes with name, cuisine, category, and ingredients.
    Use descriptive queries: ingredient names, cuisine types, or dish names (e.g. 'chicken', 'Italian pasta', 'beef stew')."""
    results = _fetch_from_mealdb(query)
    for r in results:
        _recipe_cache[r["name"]] = r
    return json.dumps(results, indent=2)


@tool
def filter_by_constraints(recipe_names: List[str], excluded_ingredients: List[str]) -> List[str]:
    """Filter recipe names to only those not containing excluded ingredients or allergens.
    Pass recipe_names as a list of names from search_recipes results.
    Pass excluded_ingredients as a list of allergens and disliked ingredients to exclude."""
    exclusions = [e.lower().strip() for e in excluded_ingredients if e and e.lower().strip() != "none"]
    if not exclusions:
        return recipe_names

    safe = []
    for name in recipe_names:
        recipe = _recipe_cache.get(name, {})
        ingredients_lower = [i.lower() for i in recipe.get("ingredients", [])]
        has_exclusion = any(
            excl in ing
            for excl in exclusions
            for ing in ingredients_lower
        )
        if not has_exclusion:
            safe.append(name)
    return safe


def parse_receipt_image(image_base64: str, media_type: str = "image/webp") -> str:
    """Use Claude vision to extract ingredient list from a base64-encoded receipt image.
    Supports image/jpeg, image/png, image/webp, image/gif."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512)
    response = llm.invoke([HumanMessage(content=[
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64,
            },
        },
        {
            "type": "text",
            "text": (
                "This is a grocery receipt. List all food items purchased as a "
                "comma-separated list of ingredient names only — no prices, quantities, or brands."
            ),
        },
    ])])
    content = response.content
    if isinstance(content, list):
        content = next(
            (b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"),
            "",
        )
    return content
